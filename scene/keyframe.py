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

from __future__ import annotations
from argparse import Namespace
import torch
import torch.nn.functional as F

from poses.feature_detector import DescribedKeypoints
from poses.triangulator import Triangulator
from scene.dense_extractor import DenseExtractor
from scene.mono_depth import MonoDepthEstimator, align_depth
from scene.optimizers import BaseAdam
from utils import sample, sixD2mtx, make_torch_sampler, depth2points
from dataloaders.read_write_model import Camera, BaseImage, rotmat2qvec


class Keyframe:
    """
    A keyframe in the scene, containing the image, camera parameters, and other information used for optimization.
    """
    def __init__(
        self,
        image: torch.Tensor,
        info: dict,
        desc_kpts: DescribedKeypoints,
        Rt: torch.Tensor,
        index: int,
        f: torch.Tensor,
        feat_extractor: DenseExtractor,
        depth_estimator: MonoDepthEstimator,
        triangulator: Triangulator,
        args: Namespace,
        inference_mode: bool = False,
    ):
        self.image_pyr = [image]
        if not inference_mode: # Only extract depth and feature maps in training mode
            self.feat_map = feat_extractor(image)
            self.mono_idepth, self.mono_depth_conf = depth_estimator(image)
            self.width = image.shape[2]
            self.height = image.shape[1]
            self.centre = torch.tensor(
                [(self.width - 1) / 2, (self.height - 1) / 2], device="cuda"
            )
            self.f = f
            self.triangulator = triangulator
            self.depth_loss_weight = args.depth_loss_weight_init
            self.depth_loss_weight_decay = args.depth_loss_weight_decay
            # Build the multiscale pyramids
            for _ in range(args.pyr_levels - 1):
                self.image_pyr.append(F.avg_pool2d(self.image_pyr[-1], 2))
            self.mask_pyr = info.pop("mask", None)
            self.pyr_lvl = args.pyr_levels - 1
            if self.mask_pyr is not None:
                self.mask_pyr = [self.mask_pyr.cuda()]
                for _ in range(args.pyr_levels - 1):
                    self.mask_pyr.append(F.avg_pool2d(self.mask_pyr[-1], 2))
                for i in range(len(self.mask_pyr)):
                    self.mask_pyr[i] = self.mask_pyr[i] > (1 - 1e-6)

        self.index = index

        self.idepth_pyr = None

        self.latest_invdepth = None
        self.desc_kpts = desc_kpts
        self.info = info
        self.is_test = info["is_test"]

        # Optimizable parameters
        self.rW2C = torch.nn.Parameter(Rt[:3, :2].clone())
        self.tW2C = torch.nn.Parameter(Rt[:3, 3].clone())
        exposure = torch.eye(3, 4, device="cuda")
        self.exposure = torch.nn.Parameter(exposure)
        self.depth_scale = torch.nn.Parameter(torch.ones(1, device="cuda"))
        self.depth_offset = torch.nn.Parameter(torch.zeros(1, device="cuda"))

        # Optimizer
        if not inference_mode: # Only create optimizer in training mode
            params = {
                "rW2C": {"val": self.rW2C, "lr": args.lr_poses},
                "tW2C": {"val": self.tW2C, "lr": args.lr_poses},
                "depth_scale": {
                    "val": self.depth_scale,
                    "lr": args.lr_depth_scale_offset,
                },
                "depth_offset": {
                    "val": self.depth_offset,
                    "lr": args.lr_depth_scale_offset,
                },
            }
            if not info["is_test"]:
                params["exposure"] = {"val": self.exposure, "lr": args.lr_exposure}
            self.optimizer = BaseAdam(params, betas=(0.8, 0.99))
            self.num_steps = 0

        self.approx_centre = -Rt[:3, :3].T @ Rt[:3, 3]

    def to(self, device: str, only_train=False):
        if self.device.type == device:
            return
        for i in range(len(self.image_pyr)):
            self.image_pyr[i] = self.image_pyr[i].to(device)
            if self.idepth_pyr is not None:
                self.idepth_pyr[i] = self.idepth_pyr[i].to(device)
            if self.mask_pyr is not None:
                self.mask_pyr[i] = self.mask_pyr[i].to(device)
        if not only_train:
            self.feat_map = self.feat_map.to(device)
            self.mono_idepth = self.mono_idepth.to(device)
            if self.latest_invdepth is not None:
                self.latest_invdepth = self.latest_invdepth.to(device)

    @property
    def device(self):
        return self.image_pyr[0].device

    def get_R(self):
        return sixD2mtx(self.rW2C)

    def get_t(self):
        return self.tW2C

    def get_Rt(self):
        Rt = torch.eye(4, device="cuda")
        Rt[:3, :3] = self.get_R()
        Rt[:3, 3] = self.get_t()
        return Rt

    def set_Rt(self, Rt: torch.Tensor):
        self.rW2C.data.copy_(Rt[:3, :2])
        self.tW2C.data.copy_(Rt[:3, 3])

        self.approx_centre = -Rt[:3, :3].T @ Rt[:3, 3]

    def get_centre(self, approx=False):
        if approx:
            return self.approx_centre
        else:
            return -self.get_R().T @ self.get_t()

    @torch.no_grad()
    def update_3dpts(self, all_keyframes: list[Keyframe]):
        """
        Assign a 3D point to each keypoint in the keyframe based on triangulation and the latest rendered depth. 
        """
        unload_desc_kpts = self.desc_kpts.kpts.device.type == "cpu"
        if unload_desc_kpts:
            self.desc_kpts.to("cuda")

        ## Update 3D points using the latest rendered depth
        if self.latest_invdepth is not None:
            uv = self.desc_kpts.kpts
            sampler = make_torch_sampler(uv.view(1, 1, -1, 2), self.width, self.height)
            model_idepth = F.grid_sample(
                self.latest_invdepth[None].cuda(),
                sampler,
                mode="bilinear",
                align_corners=True,
            )[0, 0, 0]
            mono_idepth = F.grid_sample(
                self.get_mono_idepth()[None],
                sampler,
                mode="bilinear",
                align_corners=True,
            )[0, 0, 0]
            mono_conf = F.grid_sample(
                self.mono_depth_conf, sampler, mode="bilinear", align_corners=True
            )[0, 0, 0]
            mono_model_diff = (model_idepth - mono_idepth) ** 2
            var = 0.2
            conf = 0.1 * torch.exp(-mono_model_diff / var) * mono_conf
            depth = 1 / model_idepth.clamp(1e-6, 1e6)
            new_pts = depth2points(uv, depth[..., None], self.f, self.centre)
            new_pts = (new_pts - self.get_t()) @ self.get_R()
            mask = conf > 0
            self.desc_kpts.update_3D_pts(new_pts[mask], depth[mask], conf[mask], mask)

        ## Triangulation
        # Select keyframes to triangulate with based on their locations
        uv, uvs_others, chosen_kfs_ids = self.triangulator.prepare_matches(
            self.desc_kpts
        )
        Rts_others = torch.stack(
            [all_keyframes[index].get_Rt() for i, index in enumerate(chosen_kfs_ids)],
            dim=0,
        )
        if len(Rts_others < self.triangulator.n_cams):
            Rts_others = torch.cat(
                [
                    Rts_others,
                    torch.eye(4, device="cuda")[None].repeat(
                        self.triangulator.n_cams - len(Rts_others), 1, 1
                    ),
                ],
                dim=0,
            )

        # Run the triangulator and update the 3D points
        new_pts, depth, best_dis, valid_matches = self.triangulator(
            uv, uvs_others, self.get_Rt(), Rts_others, self.f, self.centre
        )
        self.desc_kpts.update_3D_pts(
            new_pts[valid_matches], depth[valid_matches], 1, valid_matches
        )

        if unload_desc_kpts:
            self.desc_kpts.to("cpu")

    def get_mono_idepth(self, lvl=0):
        if self.idepth_pyr[lvl].device.type != "cuda":
            self.idepth_pyr[lvl] = self.idepth_pyr[lvl].cuda()
        return self.idepth_pyr[lvl] * self.depth_scale + self.depth_offset

    @torch.no_grad()
    def align_depth(self):
        """
        Align the mono depth to the triangulated depth of the keypoints.
        update_3dpts must have been called before this function.
        """
        if (self.desc_kpts.pts_conf > 0).any():
            self.mono_idepth = align_depth(
                self.mono_idepth, self.desc_kpts, self.width, self.height
            )
        self.idepth_pyr = [
            F.interpolate(
                self.mono_idepth,
                (self.height, self.width),
                mode="bilinear",
                align_corners=True,
            )[0]
        ]
        # Create the multi-scale inverse depth pyramid
        for _ in range(len(self.image_pyr) - 1):
            self.idepth_pyr.append(F.avg_pool2d(self.idepth_pyr[-1], 2))


    @torch.no_grad()
    def sample_conf(self, uv):
        return sample(
            self.mono_depth_conf, uv.view(1, 1, -1, 2), self.width, self.height
        )[0, 0, 0]

    def zero_grad(self):
        self.optimizer.zero_grad()

    @torch.no_grad()
    def step(self):
        # Optimizer step
        self.optimizer.step()
        self.depth_loss_weight *= self.depth_loss_weight_decay
        self.num_steps += 1
        # decrement pyr_lvl
        if self.num_steps % 5 == 0:
            if self.pyr_lvl > 0:
                self.image_pyr.pop()
                if self.mask_pyr is not None:
                    self.mask_pyr.pop()
                if self.idepth_pyr is not None:
                    self.idepth_pyr.pop()
                self.pyr_lvl -= 1

    def to_json(self):
        info = {
            "is_test": self.info["is_test"],
        }
        if "name" in self.info:
            info["name"] = self.info["name"]
        if "Rt" in self.info:
            info["gt_Rt"] = self.info["Rt"].cpu().numpy().tolist()

        return {
            "info": info,
            "Rt": self.get_Rt().detach().cpu().numpy().tolist(),
            "f": self.f.item(),
        }

    @classmethod
    def from_json(cls, config, index, height, width):
        if "gt_Rt" in config["info"]:
            config["info"]["Rt"] = torch.tensor(config["info"]["gt_Rt"]).cuda()
        keyframe = cls(
            image=None,
            info=config["info"],
            desc_kpts=None,
            Rt=torch.tensor(config["Rt"]).cuda(),
            index=index,
            f=None, 
            feat_extractor=None,
            depth_estimator=None,
            triangulator=None,
            args=None,
            inference_mode=True,
        )
        keyframe.height = height
        keyframe.width = width
        keyframe.centre = torch.tensor(
            [(width - 1) / 2, (height - 1) / 2], device="cuda"
        )
        return keyframe

    def to_colmap(self, id):
        """
        Convert the keyframe to a colmap camera and image.
        """
        # first param of params is focal length in pixels
        camera = Camera(
            id=id,
            model="SIMPLE_PINHOLE",
            width=self.width,
            height=self.height,
            params=[self.f.item(), self.centre[0].item(), self.centre[1].item()],
        )

        image = BaseImage(
            id=id,
            name=self.info.get("name", str(id)),
            camera_id=id,
            qvec=-rotmat2qvec(self.get_R().cpu().detach().numpy()),
            tvec=self.get_t().flatten().cpu().detach().numpy(),
            xys=[],
            point3D_ids=[],
        )

        return camera, image
