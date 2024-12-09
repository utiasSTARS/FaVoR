import numpy as np
from os import path
from tqdm import tqdm
from typing import Generator, Tuple, Union

"""
Data Loader
====================================
This module provides a base class for data loaders.
"""


class Dataloader:
    """ Base class for data loaders. """

    def __init__(self, data_path, scene):
        """
        Initializes the Dataloader with the given data path and scene name.

        Args:
            data_path (str): The path to the directory containing the data.
            scene (str): The name of the scene to load data for.
        """
        self.data_path = data_path
        self.scene = scene
        self.scene_folder = path.join(data_path, scene)
        self.gt_lines = []
        self.test_lines = []
        self.camera = None

        self.load_data()

        self.net_vlad_path = None
        self.net_vlad_poses = {}

    def load_data(self) -> None:
        """
        Placeholder method to load data for the scene.

        This method should be overridden by subclasses to implement the actual
        data loading logic.

        Returns:
            None

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        raise NotImplementedError

    def get_train(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Returns an iterator that yields training data, processed from the ground truth lines.

        This method iterates over the ground truth data and yields it after transforming
        each line using the `line2data` method. A progress bar is displayed during iteration.

        Yields:
            tuple: A processed training data containing an image and a transformation matrix.
        """
        for line in tqdm(self.gt_lines):
            yield self.line2data(line)

    def get_test(self) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Returns an iterator that yields test data, processed from the ground truth lines.

        This method iterates over the ground truth data and yields it after transforming
        each line using the `line2data` method. A progress bar is displayed during iteration.

        Yields:
            tuple: A processed test data containing an image, a ground truth transformation matrix and a prior pose.
        """
        for line in tqdm(self.test_lines):
            yield self.line2data(line, test=True)

    from typing import Union, Tuple

    def line2data(self, line: str, test: bool = False) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Transforms a data line into usable data.

        If `test` is True, additional data is returned.

        Args:
            line (str): A single line of data to be processed.
            test (bool): Flag indicating whether to process test data.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
                - For `test=False`: A tuple containing:
                    - image (ndarray): The processed image.
                    - transformation_matrix (ndarray): The transformation matrix.
                - For `test=True`: A tuple containing:
                    - image (ndarray): The processed image.
                    - transformation_matrix (ndarray): The transformation matrix.
                    - prior_transformation_matrix (ndarray): The a priori transformation matrix.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("Method not implemented.")

    def imgpath_and_pose(self, line: str) -> Tuple[str, np.ndarray]:
        """
        Extracts the image path and pose from a data line.

        Args:
            line (str): A single line of data to be processed.

        Returns:
            tuple: A tuple containing:
                - img_path (str): The path to the image.
                - cam_pose (ndarray): The camera pose.
        """
        raise NotImplementedError("Method not implemented.")

    def load_vlad(self):
        if self.net_vlad_path is None:
            raise Exception("VLAD path not set")
        net_vlad_map = {}
        with open(self.net_vlad_path, 'r') as f:
            prev_seq = None
            for line in f:
                test_seq, train_seq = line.strip().split(' ')
                if test_seq == prev_seq:
                    continue
                net_vlad_map[test_seq] = train_seq
                prev_seq = test_seq

        # load ground truth data lines:
        traing_gt_poses = {}
        for line in self.gt_lines:
            img_path, pose_train = self.imgpath_and_pose(line)
            traing_gt_poses[img_path] = pose_train

        self.net_vlad_poses = {}
        for test_seq, train_seq in net_vlad_map.items():
            self.net_vlad_poses[test_seq] = traing_gt_poses[train_seq]
