"""Rig-aware image dataset for 9-view Insta360 X5 rig.

Pass 1: iterates the reference view only, across all timesteps. This lets the
existing single-camera bootstrap + incremental pipeline run unchanged and
produce one rig pose per timestep.

Pass 2: iterates the remaining 8 views. Each frame carries a fixed relative
transform in its `info` dict; the training loop computes the view pose by
composing with the stored reference pose of the same timestep (no PnP).
"""

import json
import logging
import os
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Dict, List

import cv2
import torch

from rig.rig_loader import load_rig_config
from utils import get_image_names


def _frame_key(view: str, ts: int) -> str:
    # Unique key used both as `info["name"]` and as the dict key in `self.infos`.
    return f"{view}__ts{ts:05d}"


class RigImageDataset:
    """Two-pass iterator over a 9-view rig directory."""

    def __init__(self, args: Namespace):
        self.source_path = args.source_path
        self.images_root = os.path.join(args.source_path, args.images_dir)

        # Auto-load focal from the extraction step's metadata so a forgotten
        # --init_fov can't silently produce a wrong reconstruction. The
        # extraction script (eqr_to_pinhole.py) is the source of truth.
        meta_path = os.path.join(args.source_path, "extraction_meta.json")
        if os.path.exists(meta_path) and args.init_focal < 0 and args.init_fov < 0:
            with open(meta_path) as f:
                meta = json.load(f)
            args.init_focal = float(meta["intrinsics"]["fx"])
            print(
                f"[rig] auto-loaded init_focal={args.init_focal:.1f} "
                f"from {meta_path}"
            )

        # Load rig geometry (relative Rt per view, in COLMAP world-to-camera convention).
        self.rig = load_rig_config(args.rig_config, ref_view=args.ref_view, device="cpu")
        self.ref_view = self.rig.ref_view
        self.non_ref_views = [v for v in self.rig.view_names if v != self.ref_view]

        # Scan per-view frame files and derive the common timestep list.
        self.frames_per_view: Dict[str, List[str]] = {}
        for view in self.rig.view_names:
            view_dir = os.path.join(self.images_root, view)
            if not os.path.isdir(view_dir):
                raise FileNotFoundError(f"Rig view directory missing: {view_dir}")
            names = sorted(get_image_names(view_dir))
            self.frames_per_view[view] = names
        common = set(self.frames_per_view[self.ref_view])
        for view in self.rig.view_names:
            common &= set(self.frames_per_view[view])
        self.timestep_names = sorted(common)  # e.g. ["frame_00001.png", ...]
        if not self.timestep_names:
            raise RuntimeError("No timesteps are shared by all rig views.")
        self.ts_index = {name: i for i, name in enumerate(self.timestep_names)}
        self.num_timesteps = len(self.timestep_names)

        # Iteration order (rig Option A): every timestep emits a 9-view batch,
        # ref first, non-ref in fixed order. train.py pulls one ref frame and
        # the following 8 non-ref frames together.
        self.items: List[dict] = []
        for fname in self.timestep_names:
            ts = self.ts_index[fname]
            self.items.append({"view": self.ref_view, "ts": ts,
                               "path": os.path.join(self.images_root, self.ref_view, fname),
                               "filename": fname})
            for view in self.non_ref_views:
                self.items.append({"view": view, "ts": ts,
                                   "path": os.path.join(self.images_root, view, fname),
                                   "filename": fname})

        # start_at is in *timestep* units for rig mode (vs per-image in the
        # single-camera dataset). Drop the first N full 9-view batches and
        # shrink timestep_names accordingly so train.py's
        # `dataset.num_timesteps` reflects the remaining iterations.
        if args.start_at > 0:
            n_views_per_batch = len(self.rig.view_names)
            skip_ts = int(args.start_at)
            if skip_ts >= self.num_timesteps:
                raise ValueError(
                    f"--start_at ({skip_ts}) >= num_timesteps "
                    f"({self.num_timesteps}); nothing to iterate."
                )
            skip_items = skip_ts * n_views_per_batch
            self.items = self.items[skip_items:]
            self.timestep_names = self.timestep_names[skip_ts:]
            self.num_timesteps = len(self.timestep_names)

        # Per-item metadata consumed by the training loop.
        # is_test marks holdout-view frames so add_new_gaussians skips them
        # (pose is still estimated; only Gaussian spawn is suppressed).
        holdout_view = (getattr(args, "rig_holdout_view", "") or "").strip()
        if holdout_view and holdout_view not in self.rig.view_names:
            raise ValueError(
                f"--rig_holdout_view={holdout_view!r} is not in the rig views "
                f"{self.rig.view_names}"
            )
        if holdout_view and holdout_view == self.ref_view:
            raise ValueError(
                f"--rig_holdout_view cannot be the ref_view ({self.ref_view}); "
                "ref view is needed for bootstrap and incremental pose tracking."
            )
        self.holdout_view = holdout_view
        self.infos: Dict[str, dict] = {}
        for item in self.items:
            key = _frame_key(item["view"], item["ts"])
            self.infos[key] = {
                "is_test": (holdout_view != "" and item["view"] == holdout_view),
                "name": key,
                "rig_view": item["view"],
                "rig_ts": item["ts"],
                "rig_filename": item["filename"],
                "rig_relative_Rt": self.rig.relative_Rt[item["view"]].clone(),
            }

        # Loading config
        self.downsampling = args.downsampling
        self.num_threads = min(args.num_loader_threads, max(len(self.items), 1))
        self.current_index = 0
        self.preload_queue: Queue = Queue(maxsize=self.num_threads)
        self.executor = ThreadPoolExecutor(max_workers=self.num_threads)

        # Probe the first image for resolution (and trigger auto-downsampling if huge).
        first_image = self._load_image(self.items[0]["path"])
        self.width, self.height = first_image.shape[2], first_image.shape[1]
        res = self.width * self.height
        max_res = 1_500_000
        if self.downsampling <= 0.0 and res > max_res:
            logging.warning(
                "Large rig images, downsampling to 1.5 Mpx. "
                "Pass --downsampling=1 to disable."
            )
            self.downsampling = (res / max_res) ** 0.5
            first_image = self._load_image(self.items[0]["path"])
            self.width, self.height = first_image.shape[2], first_image.shape[1]

        self.start_preloading()

    # ---- dataset API matching ImageDataset ----

    def __len__(self) -> int:
        return len(self.items)

    def get_image_size(self):
        return self.height, self.width

    @torch.no_grad()
    def __getitem__(self, index: int):
        item = self.items[index]
        image = self._load_image(item["path"], cv2.IMREAD_UNCHANGED)
        key = _frame_key(item["view"], item["ts"])
        info = self.infos[key]
        if image.shape[0] == 4:
            info["mask"] = image[-1][None].cpu()
            image = image[:3]
        return image.cuda(), info

    def _load_image(self, image_path: str, mode: int = cv2.IMREAD_COLOR):
        image = cv2.imread(image_path, mode)
        if image is None:
            raise FileNotFoundError(f"Image at {image_path} could not be loaded.")
        if self.downsampling > 0.0 and self.downsampling != 1.0:
            image = cv2.resize(
                image, (0, 0),
                fx=1 / self.downsampling, fy=1 / self.downsampling,
                interpolation=cv2.INTER_AREA,
            )
        image = cv2.cvtColor(
            image, cv2.COLOR_BGRA2RGBA if image.shape[-1] == 4 else cv2.COLOR_BGR2RGB
        )
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image

    def _submit(self):
        if self.current_index < len(self):
            self.preload_queue.put(
                self.executor.submit(self.__getitem__, self.current_index)
            )

    def start_preloading(self):
        for self.current_index in range(self.num_threads):
            self._submit()

    def getnext(self):
        item = self.preload_queue.get().result()
        self.current_index += 1
        self._submit()
        return item
