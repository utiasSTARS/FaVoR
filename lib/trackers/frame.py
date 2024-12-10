#  Copyright (c) 2023 Vincenzo Polizzi <vincenzo dot polizzi at mail dot utoronto dot ca>
#  (Space and Terrestrial Autonomous Robotic Systems Laboratory, University of Toronto,
#  Institute for Aerospace Studies).
#  This file is subject to the terms and conditions defined in the file
#  'LICENSE', which is part of this source code package.

class Frame:
    _frame_id = 0

    def __init__(self, id: int, poses, points, patches, features, v_ids):
        """
        Initialize a Frame instance.

        Args:
            id (int): Unique ID for the frame.
            poses (list): List of pose matrices.
            points (list): List of tracked points.
            patches (list): List of image patches.
            features (list): List of feature vectors.
            v_ids (list): List of voxel IDs.
        """
        assert len(poses) == len(points) == len(patches) == len(features), \
            "Length of points, patches, features, and poses must be the same"
        self._id = id
        self.points = points
        self.pose = poses[0]
        self.patches = patches
        self.features = features
        self.v_ids = v_ids

    def __str__(self):
        return f"Frame(id={self._id}, num_points={len(self.points)})"

    def __len__(self):
        return len(self.points)

    @classmethod
    def generate_new_id(cls):
        """
        Generate and return a new unique frame ID while updating the class-level counter.

        Returns:
            int: The newly generated frame ID.
        """
        current_id = cls._frame_id
        cls._frame_id += 1
        return current_id

    def get_points(self):
        """Return the tracked points."""
        return self.points

    def get_pose(self):
        """Return the tracked pose."""
        return self.pose

    def get_patches(self):
        """Return the tracked patches."""
        return self.patches

    def get_features(self):
        """Return the tracked features."""
        return self.features

    def get_id(self):
        """Return the frame ID."""
        return self._id

    def get_v_ids(self):
        """Return the voxel IDs."""
        return self.v_ids
