import numpy as np
from os import path
from tqdm import tqdm
from typing import Generator, Tuple


class Dataloader:

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

    def get_test(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Returns an iterator that yields test data, processed from the ground truth lines.

        This method iterates over the ground truth data and yields it after transforming
        each line using the `line2data` method. A progress bar is displayed during iteration.

        Yields:
            tuple: A processed test data containing an image and a transformation matrix.
        """
        for line in tqdm(self.gt_lines):
            yield self.line2data(line)

    def line2data(self, line) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms a data line into usable data (an image and a transformation matrix).

        This method processes a data line and returns an image and a transformation matrix
        as a NumPy array. It should be implemented by subclasses to perform the actual transformation.

        Args:
            line (str): A single line of data to be processed.

        Returns:
            tuple: A tuple containing:
                - image (ndarray): The processed image.
                - transformation_matrix (ndarray): The corresponding transformation matrix.

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        raise NotImplementedError
