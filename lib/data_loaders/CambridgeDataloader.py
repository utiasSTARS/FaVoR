"""
This file contains the class CambridgeDataloader, which is a subclass of Dataloader.
"""

import numpy as np
import cv2
from os import path
from scipy.spatial.transform._rotation import Rotation
from typing import Tuple, Union

from lib.camera_models.camera import Camera
from lib.utils_favor.log_utils import print_info
from lib.data_loaders.Dataloader import Dataloader


class CambridgeDataloader(Dataloader):
    def __init__(self, data_path, scene):
        if not (scene in ['GreatCourt', 'KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch', 'Street']):
            raise Exception(
                "Scene must be one among: 'GreatCourt', 'KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch', 'Street'")
        super().__init__(data_path, scene)

        self.net_vlad_path = path.join("./datasets", 'densevlad', 'Cambridge', f'{scene}-netvlad10.txt')
        self.load_vlad()

    def load_data(self) -> None:
        # Create the camera
        focal = 0
        with open(path.join(self.scene_folder, 'reconstruction.nvm'), 'r') as f:
            reconstruction = f.readlines()
            num_cams = int(reconstruction[2])
            for i in range(3, 3 + num_cams):
                # <File name> <focal length> <quaternion WXYZ> <camera center> <radial distortion> 0
                # 0               1              2 3 4 5               6 7         8
                focal += float(reconstruction[i].split()[1])
            focal /= num_cams

        scale = 1920. / 1024.
        focal = focal / scale

        H, W = int(1080. / scale), int(1920. / scale)

        cy = H / 2.
        cx = W / 2.

        self.camera = Camera(W, H, focal, focal, cx, cy)

        # load ground truth data lines:
        ground_truth_train_path = path.join(self.scene_folder, 'dataset_train.txt')
        print_info(f"Loading training data from {ground_truth_train_path}")

        with open(ground_truth_train_path, 'r') as file:
            self.gt_lines = file.readlines()
            self.gt_lines = self.gt_lines[3:]
            file.close()

        # order lines strings
        self.gt_lines = sorted(self.gt_lines)

        # load ground truth data lines:
        ground_truth_train_path = path.join(self.scene_folder, 'dataset_test.txt')
        print_info(f"Loading training data from {ground_truth_train_path}")

        with open(ground_truth_train_path, 'r') as file:
            self.test_lines = file.readlines()
            self.test_lines = self.test_lines[3:]
            file.close()

        # order lines strings
        self.test_lines = sorted(self.test_lines)

    def imgpath_and_pose(self, line: str) -> Tuple[str, np.ndarray]:
        img_path, x, y, z, qw, qx, qy, qz = line.split()

        R = Rotation.from_quat([float(qx), float(qy), float(qz), float(qw)]).as_matrix()
        t = np.array([float(x), float(y), float(z)])
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = R
        cam_pose[:3, 3] = -R @ t

        # NOTE: in the original Cambridge dataser the pose represetnation is different from COLMAP http://ccwu.me/vsfm/doc.html#nvm
        cam_pose = np.linalg.inv(cam_pose)
        return img_path, cam_pose

    def line2data(self, line: str, test: bool = False) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        img_path, cam_pose = self.imgpath_and_pose(line)

        # store img
        img = cv2.imread(path.join(self.scene_folder, img_path))
        # check if the image exists
        if img is None:
            raise Exception(f"Image {img_path} does not exist")

        img = cv2.resize(img, (self.camera.width, self.camera.height))

        if test:
            prior_train_pose = self.net_vlad_poses[img_path]
            return img, cam_pose, prior_train_pose

        return img, cam_pose
