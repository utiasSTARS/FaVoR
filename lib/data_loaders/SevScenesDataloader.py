import numpy as np
import cv2
from os import path
from scipy.spatial.transform._rotation import Rotation
from typing import Tuple, Union

from lib.camera_models.camera import Camera
from lib.utils_favor.log_utils import print_info
from lib.data_loaders.Dataloader import Dataloader


class SevScenesDataloader(Dataloader):

    def __init__(self, data_path, scene):
        if not (scene in ['chess', 'pumpkin', 'fire', 'heads', 'office', 'redkitchen', 'stairs']):
            raise Exception(
                "Scene must be one among: 'chess', 'pumpkin', 'fire', 'heads', 'office', 'redkitchen', 'stairs'")
        super().__init__(data_path, scene)
        self.net_vlad_path = path.join("./datasets", 'densevlad', '7-Scenes', f'{self.scene}_top10.txt')


    def load_data(self) -> None:
        # Create the camera
        ground_truth_train_path = path.join(self.data_path, "COLMAP_gt", self.scene + "_train.txt")

        # read the focal length from the colmap estimation
        focal = 0
        count = 0
        with open(ground_truth_train_path, 'r') as file:
            for line in file:
                _, _, _, _, _, _, _, _, f = line.split()
                focal += float(f)
                count += 1

        # get the average focal length
        focal /= count

        H, W = 480, 640

        cy = H / 2.
        cx = W / 2.

        self.camera = Camera(W, H, focal, focal, cx, cy)

        # load ground truth data lines:
        print_info(f"Loading training data from {ground_truth_train_path}")
        with open(ground_truth_train_path, 'r') as file:
            self.gt_lines = file.readlines()
            file.close()

        # order lines strings
        self.gt_lines = sorted(self.gt_lines)

        ground_truth_train_path = path.join(self.data_path, "COLMAP_gt", self.scene + "_test.txt")
        print_info(f"Loading test data from {ground_truth_train_path}")
        with open(ground_truth_train_path, 'r') as file:
            self.test_lines = file.readlines()
            file.close()

        # order lines strings
        self.test_lines = sorted(self.test_lines)

    def imgpath_and_pose(self, line: str) -> Tuple[str, np.ndarray]:
        img_path, qw, qx, qy, qz, x, y, z, _ = line.split()

        R = Rotation.from_quat([float(qx), float(qy), float(qz), float(qw)]).as_matrix()
        t = np.array([float(x), float(y), float(z)])
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = R
        cam_pose[:3, 3] = t
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

        if test:
            prior_train_pose = self.net_vlad_poses[img_path]
            return img, cam_pose, prior_train_pose

        return img, cam_pose


