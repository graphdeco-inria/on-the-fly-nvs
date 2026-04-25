#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import time

import numpy as np
import torch
from tqdm import tqdm

from socketserver import TCPServer
from http.server import SimpleHTTPRequestHandler
from args import get_args
from threading import Thread
from dataloaders.image_dataset import ImageDataset
from dataloaders.rig_image_dataset import RigImageDataset
from dataloaders.stream_dataset import StreamDataset
from poses.feature_detector import Detector
from poses.matcher import Matcher
from poses.pose_initializer import PoseInitializer
from poses.triangulator import Triangulator
from scene.dense_extractor import DenseExtractor
from scene.keyframe import Keyframe
from scene.mono_depth import MonoDepthEstimator
from scene.scene_model import SceneModel
from gaussianviewer import GaussianViewer
from webviewer.webviewer import WebViewer
from graphdecoviewer.types import ViewerMode
from utils import align_mean_up_fwd, increment_runtime

if __name__ == "__main__":
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    args = get_args()

    # Initialize dataloader
    if "://" in args.source_path:
        dataset = StreamDataset(args.source_path, args.downsampling)
        is_stream = True
    elif args.use_rig:
        dataset = RigImageDataset(args)
        is_stream = False
    else:
        dataset = ImageDataset(args)
        is_stream = False
    height, width = dataset.get_image_size()

    # Initialize other modules
    print("Initializing modules and running just in time compilation, may take a while...")
    max_error = max(args.match_max_error * width, 1.5)
    min_displacement = max(args.min_displacement * width, 30)
    matcher = Matcher(args.fundmat_samples, max_error)
    triangulator = Triangulator(
        args.num_kpts, args.num_prev_keyframes_miniba_incr, max_error
    )
    pose_initializer = PoseInitializer(
        width, height, triangulator, matcher, 2 * max_error, args,
        rig_config=(dataset.rig if args.use_rig else None),
    )
    focal = pose_initializer.f_init
    dense_extractor = DenseExtractor(width, height)
    depth_estimator = MonoDepthEstimator(width, height)
    scene_model = SceneModel(width, height, args, matcher)
    detector = Detector(args.num_kpts, width, height)

    # Initialize the viewer
    if args.viewer_mode in ["server", "local"]:
        viewer_mode = ViewerMode.SERVER if args.viewer_mode == "server" else ViewerMode.LOCAL
        viewer = GaussianViewer.from_scene_model(scene_model, viewer_mode)
        viewer_thd = Thread(target=viewer.run, args=(args.ip, args.port), daemon=True)
        viewer_thd.start()
        viewer.throttling = True # Enable throttling when training
    elif args.viewer_mode == "web":
        ip = "0.0.0.0"
        server = TCPServer((ip, 8000), SimpleHTTPRequestHandler)
        server_thd = Thread(target=server.serve_forever, daemon=True)
        server_thd.start()
        print(f"Visit http://{ip}:8000/webviewer to for the viewer")

        viewer = WebViewer(scene_model, args.ip, args.port)
        viewer_thd = Thread(target=viewer.run, daemon=True)
        viewer_thd.start()

    n_active_keyframes = 0
    n_keyframes = 0
    needs_reboot = False
    bootstrap_keyframe_dicts = []
    bootstrap_desc_kpts = []
    # Rig mode (Option A): one 9-view batch per timestep; the first B batches
    # accumulate for rig bootstrap, then each subsequent batch is handled by
    # rig-aware incremental PnP. `ref_kf_by_ts[ts]` caches the ref-view keyframe
    # for debugging / downstream inspection.
    ref_kf_by_ts = {}
    bootstrap_rig_data = []
    n_rig_bootstrap_ts = 0

    # Dict of runtimes for each step
    runtimes = ["Load", "BAB", "tri", "BAI", "Add", "Init", "Opt", "anc"]
    runtimes = {key: [0, 0] for key in runtimes}
    metrics = {}

    ## Scene reconstruction
    print(f"Starting reconstruction for {args.source_path}")
    # In rig mode each loop iteration consumes 9 images (one 9-view batch),
    # so the tqdm range must be in batches rather than individual images.
    total_iters = (
        dataset.num_timesteps if args.use_rig else len(dataset)
    )
    pbar = tqdm(range(0, total_iters))
    reconstruction_start_time = time.time()
    for frameID in pbar:
        start_time = time.time()

        if args.viewer_mode == "web":
            viewer.trainer_state = "running"

            # Paused
            while viewer.state == "stop":
                pbar.set_postfix_str(
                    "\033[31mPaused. Press the Start button in the webviewer\033[0m"
                )
                time.sleep(0.1)
            
            # Finish reconstruction
            if viewer.state == "finish":
                viewer.trainer_state = "finish"
                break
        
        # === Rig mode (Option A): one 9-view batch per timestep ===
        if args.use_rig:
            image, info = dataset.getnext()
            if info["rig_view"] != dataset.ref_view:
                raise RuntimeError(
                    "Rig dataset desync: expected a ref-view frame at batch "
                    f"start, got view={info['rig_view']} ts={info['rig_ts']}"
                )
            ts = info["rig_ts"]
            rig_batch = {dataset.ref_view: (image, info, detector(image))}
            for _ in range(len(dataset.non_ref_views)):
                nr_img, nr_info = dataset.getnext()
                if nr_info["rig_ts"] != ts:
                    raise RuntimeError(
                        f"rig batch desync: expected ts={ts}, got ts={nr_info['rig_ts']}"
                    )
                rig_batch[nr_info["rig_view"]] = (nr_img, nr_info, detector(nr_img))
            increment_runtime(runtimes["Load"], start_time)

            B = args.num_keyframes_miniba_bootstrap
            view_order = list(dataset.rig.view_names)

            if n_rig_bootstrap_ts < B:
                bootstrap_rig_data.append({"ts": ts, "frames": rig_batch})
                n_rig_bootstrap_ts += 1
                if n_rig_bootstrap_ts < B:
                    continue

                # --- Rig bootstrap: run MiniBARig over B timesteps * 9 views ---
                start_time = time.time()
                desc_per_ts_per_view = [
                    {v: data["frames"][v][2] for v in view_order}
                    for data in bootstrap_rig_data
                ]
                rig_Rts, f_out, res, _xyz, _view_names = (
                    pose_initializer.initialize_bootstrap_rig(
                        desc_per_ts_per_view, dataset.rig
                    )
                )
                focal = f_out.cpu().item()
                increment_runtime(runtimes["BAB"], start_time)

                f_tensor = torch.tensor([focal], device="cuda", dtype=torch.float32)
                ref_kf_scene_indices = []  # indices (into scene_model.keyframes) of ref keyframes
                start_time = time.time()
                for ts_i, data in enumerate(bootstrap_rig_data):
                    rig_pose = rig_Rts[ts_i]
                    for v_name in view_order:
                        img, inf, desc = data["frames"][v_name]
                        rel = inf["rig_relative_Rt"].to("cuda")
                        Rt_view = rel @ rig_pose
                        kf = Keyframe(
                            img, inf, desc, Rt_view, n_keyframes, f_tensor,
                            dense_extractor, depth_estimator, triangulator, args,
                        )
                        # Push the focal into scene_model on the very first keyframe.
                        first = (ts_i == 0 and v_name == view_order[0])
                        scene_model.add_keyframe(kf, f_tensor if first else None)
                        if v_name == dataset.ref_view:
                            ref_kf_by_ts[data["ts"]] = kf
                            ref_kf_scene_indices.append(len(scene_model.keyframes) - 1)
                        prev_keyframe = kf
                        n_keyframes += 1
                increment_runtime(runtimes["Add"], start_time)

                # Gaussian initialization from ALL 9 views (was: ref only).
                # Each view's desc_kpts.pts3d is filled in by update_3dpts()
                # inside add_new_gaussians via its own time-axis matches that
                # bootstrap_rig has already generated. add_new_gaussians also
                # handles align_depth internally, so the explicit non-ref
                # align_depth loop is no longer needed.
                start_time = time.time()
                first_bootstrap_scene_idx = (
                    len(scene_model.keyframes)
                    - len(bootstrap_rig_data) * len(view_order)
                )
                for scene_idx in range(first_bootstrap_scene_idx, len(scene_model.keyframes)):
                    scene_model.add_new_gaussians(scene_idx)
                increment_runtime(runtimes["Init"], start_time)

                start_time = time.time()
                if is_stream:
                    scene_model.optimize_async(args.num_iterations)
                else:
                    scene_model.optimization_loop(args.num_iterations)
                increment_runtime(runtimes["Opt"], start_time)
                last_reboot = n_keyframes
                if args.viewer_mode not in ["none", "web"]:
                    viewer.reset_intrinsics("point_view")
                continue

            # --- Rig-incremental (post-bootstrap) ---
            start_time = time.time()
            view_indices = {v: n_keyframes + i for i, v in enumerate(view_order)}
            desc_per_view = {v: rig_batch[v][2] for v in view_order}
            # Per-view prev_keyframes: each view picks its own candidate pool
            # based on its own features. Critical for U-turn handling, where
            # a single view's best prev frames are very different from ref's.
            # update_3dpts=True only on the first call to avoid redundant
            # triangulation across the 9 per-view queries.
            prev_per_view = {}
            for i, v in enumerate(view_order):
                prev_per_view[v] = scene_model.get_prev_keyframes(
                    args.num_prev_keyframes_miniba_incr, i == 0, desc_per_view[v],
                )
            increment_runtime(runtimes["tri"], start_time)

            start_time = time.time()
            rig_pose, _ = pose_initializer.initialize_incremental_rig(
                prev_per_view, desc_per_view, view_indices, dataset.rig,
            )
            increment_runtime(runtimes["BAI"], start_time)
            if rig_pose is None:
                continue

            f_tensor = torch.tensor([focal], device="cuda", dtype=torch.float32)
            start_time = time.time()
            for v_name in view_order:
                img, inf, desc = rig_batch[v_name]
                rel = inf["rig_relative_Rt"].to("cuda")
                Rt_view = rel @ rig_pose
                kf = Keyframe(
                    img, inf, desc, Rt_view, n_keyframes, f_tensor,
                    dense_extractor, depth_estimator, triangulator, args,
                )
                scene_model.add_keyframe(kf)
                prev_keyframe = kf
                if v_name == dataset.ref_view:
                    ref_kf_by_ts[ts] = kf
                # spawn from all 9 views (was: ref only). add_new_gaussians
                # handles align_depth + update_3dpts internally.
                scene_model.add_new_gaussians()
                n_keyframes += 1
            increment_runtime(runtimes["Add"], start_time)
            start_time = time.time()
            if is_stream:
                scene_model.optimize_async(args.num_iterations)
            else:
                scene_model.optimization_loop(args.num_iterations)
            increment_runtime(runtimes["Opt"], start_time)
            start_time = time.time()
            scene_model.place_anchor_if_needed()
            increment_runtime(runtimes["anc"], start_time)
            continue

        # === Non-rig path ===
        if n_keyframes == 0:
            image, info = dataset.getnext()
            prev_desc_kpts = detector(image)
            bootstrap_keyframe_dicts = [{"image": image, "info": info}]
            bootstrap_desc_kpts = [prev_desc_kpts]
            n_keyframes += 1
            continue

        image, info = dataset.getnext()

        desc_kpts = detector(image)
        # Match features between the previous and current frame
        curr_prev_matches = matcher(desc_kpts, prev_desc_kpts)
        # Determine if we should add a keyframe based on the matches
        dist = torch.norm(curr_prev_matches.kpts - curr_prev_matches.kpts_other, dim=-1)
        should_add_keyframe = (
            dist.median() > min_displacement
            and len(curr_prev_matches.kpts) > args.min_num_inliers
        )
        # Always add test frames so we estimate their poses
        should_add_keyframe |= info["is_test"]
        increment_runtime(runtimes["Load"], start_time)

        if should_add_keyframe:
            ## Bootstrap
            # Accumulate keyframes for pose initialization
            if n_keyframes < args.num_keyframes_miniba_bootstrap:
                bootstrap_keyframe_dicts.append({"image": image, "info": info})
                bootstrap_desc_kpts.append(desc_kpts)

            if n_keyframes == args.num_keyframes_miniba_bootstrap - 1:
                start_time = time.time()
                Rts, f, _ = pose_initializer.initialize_bootstrap(bootstrap_desc_kpts)
                focal = f.cpu().item()
                increment_runtime(runtimes["BAB"], start_time)
                for index, (keyframe_dict, desc_kpts, Rt) in enumerate(
                    zip(bootstrap_keyframe_dicts, bootstrap_desc_kpts, Rts)
                ):
                    start_time = time.time()
                    if args.use_colmap_poses:
                        Rt = keyframe_dict["info"]["Rt"]
                        f = keyframe_dict["info"]["focal"]
                    keyframe = Keyframe(
                        keyframe_dict["image"],
                        keyframe_dict["info"],
                        desc_kpts,
                        Rt,
                        index,
                        f,
                        dense_extractor,
                        depth_estimator,
                        triangulator,
                        args,
                    )
                    scene_model.add_keyframe(keyframe, f)
                    increment_runtime(runtimes["Add"], start_time)
                if args.viewer_mode not in ["none", "web"]:
                    viewer.reset_intrinsics("point_view")
                prev_keyframe = keyframe
                for index in range(args.num_keyframes_miniba_bootstrap):
                    start_time = time.time()
                    scene_model.add_new_gaussians(index)
                    increment_runtime(runtimes["Init"], start_time)
                start_time = time.time()
                # Run initial optimization on the bootstrap keyframes
                # If streaming, run async optimization until the next keyframe is added
                if is_stream:
                    scene_model.optimize_async(args.num_iterations)
                else:
                    scene_model.optimization_loop(args.num_iterations)
                increment_runtime(runtimes["Opt"], start_time)
                last_reboot = n_keyframes

            ## Reboot
            if (
                args.enable_reboot
                and scene_model.approx_cam_centres is not None
                and len(scene_model.anchors)
            ):
                # Check if the camera baseline is a lot smaller or larger than expected
                last_centers = scene_model.approx_cam_centres[-20:]
                rel_dist = torch.norm(
                    last_centers[1:] - last_centers[:-1], dim=-1
                ).mean()
                needs_reboot = (
                    rel_dist > 0.1 * 5 or rel_dist < 0.1 / 3
                ) and n_keyframes - last_reboot > 50
            if needs_reboot:
                # Reboot: run mini BA on the last 8 keyframes
                bs_kfs = scene_model.keyframes[-8:]
                bootstrap_desc_kpts = [bs_kf.desc_kpts for bs_kf in bs_kfs]
                in_Rts = torch.stack([kf.get_Rt() for kf in bs_kfs])
                Rts, _, final_residual = pose_initializer.initialize_bootstrap(
                    bootstrap_desc_kpts, rebooting=True
                )
                # Check if the reboot succeeded
                if final_residual < max_error * 0.5:
                    Rts = align_mean_up_fwd(Rts, in_Rts)
                    for Rt, keyframe in zip(Rts, bs_kfs):
                        keyframe.set_Rt(Rt)
                    # Reset the scene model and reinitialize the gaussians
                    scene_model.reset()
                    for i in range(3, 0, -1):
                        scene_model.add_new_gaussians(-i)
                    for _ in range(3 * args.num_iterations):
                        scene_model.optimization_step()
                    needs_reboot = False
                    last_reboot = n_keyframes

            ## Incremental reconstruction
            # Incremental pose initialization
            if n_keyframes >= args.num_keyframes_miniba_bootstrap:
                start_time = time.time()
                prev_keyframes = scene_model.get_prev_keyframes(
                    args.num_prev_keyframes_miniba_incr, True, desc_kpts
                )
                increment_runtime(runtimes["tri"], start_time)
                start_time = time.time()
                Rt = pose_initializer.initialize_incremental(
                    prev_keyframes, desc_kpts, n_keyframes, info["is_test"], image
                )
                increment_runtime(runtimes["BAI"], start_time)
                start_time = time.time()
                if Rt is not None:
                    if args.use_colmap_poses:
                        Rt = info["Rt"]
                    keyframe = Keyframe(
                        image,
                        info,
                        desc_kpts,
                        Rt,
                        n_keyframes,
                        f,
                        dense_extractor,
                        depth_estimator,
                        triangulator,
                        args,
                    )
                    scene_model.add_keyframe(keyframe)
                    prev_keyframe = keyframe
                    increment_runtime(runtimes["Add"], start_time)
                    # Gaussian initialization
                    start_time = time.time()
                    scene_model.add_new_gaussians()
                    increment_runtime(runtimes["Init"], start_time)
                    start_time = time.time()
                    # If streaming, run async optimization until the next keyframe is added
                    if is_stream:
                        scene_model.optimize_async(args.num_iterations)
                    else:
                        scene_model.optimization_loop(args.num_iterations)
                    increment_runtime(runtimes["Opt"], start_time)
                else:
                    should_add_keyframe = False

        if should_add_keyframe:
            ## Check if anchor creation is needed based on the primitives' size 
            start_time = time.time()
            scene_model.place_anchor_if_needed()
            increment_runtime(runtimes["anc"], start_time)

            n_keyframes += 1
            if not info["is_test"]:
                prev_desc_kpts = desc_kpts

            ## Intermediate evaluation
            if (
                n_keyframes % args.test_frequency == 0
                and args.test_frequency > 0
                and (args.test_hold > 0 or args.eval_poses)
            ):
                metrics = scene_model.evaluate(args.eval_poses)

            ## Save intermediate model
            if (
                frameID % args.save_every == 0
                and args.save_every > 0
            ):
                scene_model.save(
                    os.path.join(args.model_path, "progress", f"{frameID:05d}")
                )

            ## Display optimization progress and metrics
            bar_postfix = []
            for key, value in metrics.items():
                bar_postfix += [f"\033[31m{key}:{value:.2f}\033[0m"]
            if args.display_runtimes:
                for key, value in runtimes.items():
                    if value[1] > 0:
                        bar_postfix += [
                            f"\033[35m{key}:{1000 * value[0] / value[1]:.1f}\033[0m"
                        ]
            bar_postfix += [
                f"\033[36mFocal:{focal:.1f}",
                f"\033[36mKeyframes:{n_keyframes}\033[0m",
                f"\033[36mGaussians:{scene_model.n_active_gaussians}\033[0m",
                f"\033[36mAnchors:{len(scene_model.anchors)}\033[0m",
            ]
            pbar.set_postfix_str(",".join(bar_postfix), refresh=False)

    reconstruction_time = time.time() - reconstruction_start_time

    # Set to inference mode so that the model can be rendered properly
    scene_model.enable_inference_mode()

    # === Post-hoc render evaluation (all keyframes) =========================
    # Motivated by rig-mode having no test_hold: render each keyframe from its
    # own pose, compare against the loaded GT, dump images + per-frame metrics.
    # Backward is never called here → bypasses the iter≥10 CUDA rasterizer
    # crash and works with iter=2 baseline.
    if getattr(args, "use_rig", False):
        import cv2
        import numpy as np
        from fused_ssim import fused_ssim
        from utils import psnr as psnr_fn
        try:
            import lpips as lpips_mod
            lpips_fn = lpips_mod.LPIPS(net="vgg").cuda().eval()
        except Exception:
            lpips_fn = None
        eval_dir = os.path.join(args.model_path, "render_eval")
        os.makedirs(eval_dir, exist_ok=True)
        per_frame = []
        with torch.no_grad():
            for kf in scene_model.keyframes:
                pkg = scene_model.render_from_id(kf.index, pyr_lvl=0)
                rendered = pkg["render"].clamp(0, 1)
                gt = kf.image_pyr[0].to(rendered.device)
                if gt.shape[-2:] != rendered.shape[-2:]:
                    continue
                p = float(psnr_fn(rendered, gt))
                s = float(fused_ssim(rendered[None], gt[None], train=False).item())
                if lpips_fn is not None:
                    r01 = rendered[None] * 2 - 1
                    g01 = gt[None] * 2 - 1
                    l = float(lpips_fn(r01, g01).item())
                else:
                    l = float("nan")
                name = kf.info.get("name", f"kf{kf.index:04d}")
                if not name.endswith(".png"):
                    name = name + ".png"
                # side-by-side: rendered | gt
                img_np = torch.cat([rendered, gt], dim=-1)
                img_np = (img_np.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(eval_dir, name), img_np)
                per_frame.append({
                    "name": name, "rig_ts": kf.info.get("rig_ts"),
                    "rig_view": kf.info.get("rig_view"),
                    "is_test": bool(kf.info.get("is_test", False)),
                    "psnr": p, "ssim": s, "lpips": l,
                })
        if per_frame:
            psnrs = [x["psnr"] for x in per_frame]
            ssims = [x["ssim"] for x in per_frame]
            lpipss = [x["lpips"] for x in per_frame if not (x["lpips"] != x["lpips"])]
            summary = {
                "num_frames": len(per_frame),
                "psnr_mean": float(np.mean(psnrs)),
                "ssim_mean": float(np.mean(ssims)),
                "lpips_mean": (float(np.mean(lpipss)) if lpipss else float("nan")),
                "psnr_min": float(np.min(psnrs)),
                "psnr_max": float(np.max(psnrs)),
            }
            import json as _json
            with open(os.path.join(eval_dir, "metrics.json"), "w") as _f:
                _json.dump({"summary": summary, "per_frame": per_frame}, _f, indent=2)
            print(
                "[post-hoc render] "
                f"n={summary['num_frames']}  "
                f"PSNR={summary['psnr_mean']:.2f}  "
                f"SSIM={summary['ssim_mean']:.3f}  "
                f"LPIPS={summary['lpips_mean']:.3f}  "
                f"(range PSNR {summary['psnr_min']:.2f}–{summary['psnr_max']:.2f})"
            )
            # Honest eval: when --rig_holdout_view is set, split metrics
            # into train (views included in optimization) vs holdout
            # (excluded — provides a real generalization signal).
            holdout_pf = [x for x in per_frame if x["is_test"]]
            train_pf = [x for x in per_frame if not x["is_test"]]
            if holdout_pf:
                hp = [x["psnr"] for x in holdout_pf]
                hs = [x["ssim"] for x in holdout_pf]
                hl = [x["lpips"] for x in holdout_pf
                      if not (x["lpips"] != x["lpips"])]
                tp = [x["psnr"] for x in train_pf]
                ts = [x["ssim"] for x in train_pf]
                tl = [x["lpips"] for x in train_pf
                      if not (x["lpips"] != x["lpips"])]
                print(
                    f"[holdout view={args.rig_holdout_view}] "
                    f"n={len(holdout_pf)}  "
                    f"PSNR={float(np.mean(hp)):.2f}  "
                    f"SSIM={float(np.mean(hs)):.3f}  "
                    f"LPIPS={(float(np.mean(hl)) if hl else float('nan')):.3f}"
                )
                print(
                    f"[train views] "
                    f"n={len(train_pf)}  "
                    f"PSNR={float(np.mean(tp)):.2f}  "
                    f"SSIM={float(np.mean(ts)):.3f}  "
                    f"LPIPS={(float(np.mean(tl)) if tl else float('nan')):.3f}"
                )
    # ========================================================================

    # Save the model and metrics
    print("Saving the reconstruction to:", args.model_path)
    metrics = scene_model.save(args.model_path, reconstruction_time, len(dataset))
    print(
        ", ".join(
            f"{metric}: {value:.3f}"
            if isinstance(value, float)
            else f"{metric}: {value}"
            for metric, value in metrics.items()
        )
    )

    # Fine tuning after initial reconstruction
    if len(args.save_at_finetune_epoch) > 0:
        finetune_epochs = max(args.save_at_finetune_epoch)
        torch.cuda.empty_cache()
        scene_model.inference_mode = False
        pbar = tqdm(range(0, finetune_epochs), desc="Fine tuning")
        for epoch in pbar:
            # Run one epoch of fine-tuning
            epoch_start_time = time.time()
            scene_model.finetune_epoch()
            epoch_time = time.time() - epoch_start_time
            reconstruction_time += epoch_time
            # Save the model and metrics
            if epoch + 1 in args.save_at_finetune_epoch:
                torch.cuda.empty_cache()
                scene_model.inference_mode = True
                metrics = scene_model.save(
                    os.path.join(args.model_path, str(epoch + 1)), reconstruction_time
                )
                bar_postfix = []
                for key, value in metrics.items():
                    bar_postfix += [f"\033[31m{key}:{value:.2f}\033[0m"]
                pbar.set_postfix_str(",".join(bar_postfix))
                scene_model.inference_mode = False
                torch.cuda.empty_cache()
                
        # Set to inference mode so that the model can be rendered properly
        scene_model.inference_mode = True

    if args.viewer_mode != "none":
        if args.viewer_mode == "web":
            while True:
                time.sleep(1)
        else:
            viewer.throttling = False # Disable throttling when done training
            # Loop to keep the viewer alive
            while viewer.running:
                time.sleep(1)
