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

import cv2
import os
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import logging
from argparse import Namespace

from dataloaders.read_write_model import read_model, qvec2rotmat
from utils import get_image_names


class ImageDataset:
    """
    The main dataset class for loading images from disk in a multithreaded manner.
    It also supports loading masks and COLMAP poses if available.
    The next image can be fetched using the `getnext` method.
    """
    def __init__(self, args: Namespace):
        self.images_dir = os.path.join(args.source_path, args.images_dir)
        self.image_name_list = get_image_names(self.images_dir)
        self.image_name_list.sort()
        self.image_name_list = self.image_name_list[args.start_at :]
        self.image_paths = [
            os.path.join(self.images_dir, image_name)
            for image_name in self.image_name_list
        ]
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

        self.mask_dir = (
            os.path.join(args.source_path, args.masks_dir) if args.masks_dir else None
        )
        if self.mask_dir:
            self.mask_paths = [
                os.path.join(self.mask_dir, os.path.splitext(image_name)[0] + ".png")
                for image_name in self.image_name_list
            ]
            assert all(os.path.exists(mask_path) for mask_path in self.mask_paths), (
                "Not all masks exist."
            )

        self.downsampling = args.downsampling
        self.num_threads = min(args.num_loader_threads, len(self.image_paths))
        self.current_index = 0
        self.preload_queue = Queue(maxsize=self.num_threads)
        self.executor = ThreadPoolExecutor(max_workers=self.num_threads)

        self.infos = {
            name: {
                "is_test": (args.test_hold > 0) and (i % args.test_hold == 0),
                "name": name,
            }
            for i, name in enumerate(self.image_name_list)
        }

        first_image = self._load_image(self.image_paths[0])
        self.width, self.height = first_image.shape[2], first_image.shape[1]
        res = self.width * self.height
        max_res = 1_500_000  # 1.5 Mpx
        if self.downsampling <= 0.0 and res > max_res:
            logging.warning(
                "Large images, downsampling to 1.5 Mpx. "
                "If this is not desired, please use --downsampling=1"
            )
            self.downsampling = (res / max_res) ** 0.5
            first_image = self._load_image(self.image_paths[0])
            self.width, self.height = first_image.shape[2], first_image.shape[1]


        # Load COLMAP data
        self.load_colmap_data(os.path.join(args.source_path, "sparse/0"))

        # Check that all images have poses
        has_all_poses = all(
            "Rt" in self.infos[image_name] for image_name in self.image_name_list
        )
        if args.use_colmap_poses:
            assert has_all_poses, (
                "COLMAP poses are required but not all images have poses."
            )
            self.align_colmap_poses()

        if args.eval_poses and not has_all_poses:
            logging.warning(
                " Not all images have COLMAP poses, pose evaluation will be skipped."
            )

        self.start_preloading()

    def __len__(self):
        return len(self.image_paths)

    @torch.no_grad()
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = self._load_image(image_path, cv2.IMREAD_UNCHANGED)
        info = self.infos[os.path.basename(image_path)]
        if image.shape[0] == 4:
            info["mask"] = image[-1][None].cpu()
            image = image[:3]
        if self.mask_dir:
            mask = self._load_image(self.mask_paths[index])
            info["mask"] = mask[0][None]
        return image.cuda(), info

    def _load_image(self, image_path, mode=cv2.IMREAD_COLOR):
        image = cv2.imread(image_path, mode)
        if image is None:
            raise FileNotFoundError(f"Image at {image_path} could not be loaded.")
        if self.downsampling > 0.0 and self.downsampling != 1.0:
            image = cv2.resize(
                image,
                (0, 0),
                fx=1 / self.downsampling,
                fy=1 / self.downsampling,
                interpolation=cv2.INTER_AREA,
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA if image.shape[-1] == 4 else cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image

    def _submit(self):
        if self.current_index < len(self):
            self.preload_queue.put(
                self.executor.submit(self.__getitem__, self.current_index)
            )

    def start_preloading(self):
        """Start threads to preload images."""
        for self.current_index in range(self.num_threads):
            self._submit()

    def getnext(self):
        """Get the next item from the dataset and start preloading the next one."""
        item = self.preload_queue.get().result()
        self.current_index += 1
        self._submit()
        return item

    def get_image_size(self):
        return self.height, self.width

    def load_colmap_data(self, colmap_folder_path):
        """Load COLMAP camera intrinsics and extrinsics. Stores them in self.infos."""
        try:
            cameras, images, _ = read_model(colmap_folder_path)
        except Exception as e:
            logging.warning(
                f" Failed to read COLMAP files in {colmap_folder_path}: {e}"
            )
            return
        if len(cameras) != 1:
            logging.warning(" Only supports one camera")
        model = list(cameras.values())[0].model
        if model != "PINHOLE" and model != "SIMPLE_PINHOLE":
            logging.warning(" Unexpected camera model: " + model)

        for image_id, image in images.items():
            camera = cameras[image.camera_id]

            # Intrinsics and projection matrix
            focal_x = camera.params[0]
            focal_y = camera.params[1] if camera.model == "PINHOLE" else focal_x
            focal = (focal_x + focal_y) / 2
            focal = focal_x * self.width / camera.width

            # Pose
            Rt = np.eye(4, dtype=np.float32)
            Rt[:3, :3] = qvec2rotmat(image.qvec)
            Rt[:3, 3] = image.tvec

            # Store CameraInfo for each image
            name = os.path.basename(image.name)
            if image.name in self.infos:
                self.infos[name]["Rt"] = torch.tensor(Rt, device="cuda")
                self.infos[name]["focal"] = torch.tensor([focal], device="cuda").float()

    def align_colmap_poses(self):
        """Scale and set first Rt as identity"""
        centres = []
        for idx in range(6):
            centres.append(self.infos[self.image_name_list[idx]]["Rt"].inverse()[:3, 3])
        centres = torch.stack(centres)
        rel_ts = centres[:-1] - centres[1:]

        scale = 0.1 / rel_ts.norm(dim=-1).mean()
        inv_first_Rt = self.infos[self.image_name_list[0]]["Rt"].inverse()
        for info in self.infos.values():
            info["Rt"] = info["Rt"] @ inv_first_Rt
            info["Rt"][:3, 3] *= scale
