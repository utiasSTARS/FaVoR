#  Copyright (c) 2023 Vincenzo Polizzi <vincenzo dot polizzi at mail dot utoronto dot ca>
#  (Space and Terrestrial Autonomous Robotic Systems Laboratory, University of Toronto,
#  Institute for Aerospace Studies).
#  This file is subject to the terms and conditions defined in the file
#  'LICENSE', which is part of this source code package.
import numpy as np
import torch
from einops import rearrange


class Track:
    _track_id = 0  # Static counter for track IDs

    @classmethod
    def generate_new_id(cls):
        """
        Generate and return a new unique track ID while updating the class-level counter.

        Returns:
            int: The newly generated track ID.
        """
        current_id = cls._track_id
        cls._track_id += 1
        return current_id

    def __init__(self, patch_size_half=3, distortion=None, id_=None):
        """
        Initialize a Track instance.

        Args:
            patch_size_half (int): Half of the patch size (for feature extraction).
            distortion: Distortion correction model (optional).
            id_ (int, optional): Manually specified track ID. If None, a new ID is generated.
        """
        self._id = id_ if id_ is not None else self.generate_new_id()
        self.points = []
        self.poses = []
        self.point_w = np.array([0, 0, 0])
        self.patch_size_half = patch_size_half
        self._is_dead = True
        self.frames_id = []
        self.features = []
        self.patches = []
        self.last_frame_id = -1

        # For training purposes
        self.near = -1.0
        self.far = -1.0
        self.xyz_min = -1.0
        self.xyz_max = -1.0

        self.distortion = distortion
        self.descriptors_list = []
        self.track_descriptor = None

    def __str__(self):
        return f"Track(id={self._id}, length={len(self.points)}, is_dead={self.is_dead()})"

    def __len__(self):
        return len(self.points)

    def is_dead(self):
        """Check if the track is marked as dead."""
        return self._is_dead

    def kill(self):
        """Mark the track as dead."""
        self._is_dead = True

    def set_alive(self):
        """Mark the track as alive."""
        self._is_dead = False

    def append(self, point, pose, feat, patch, frame_id, prev_point=None, descriptor=None):
        """
        Append a new point, pose, feature, and patch to the track.

        Args:
            point: The new 2D/3D point to add.
            pose: The corresponding camera pose.
            feat: The feature vector.
            patch: The patch image.
            frame_id: The frame ID associated with this point.
            prev_point: The previous point (optional).
            descriptor: Custom descriptor (optional).

        Returns:
            bool: True if the point is added, False otherwise.
        """
        if self.last_frame_id == frame_id:
            return False

        if prev_point is None or self._is_same_point(prev_point):
            if self.distortion:
                point = self.distortion.undistort(point.reshape((1, 2)))[0]
            self.points.append(point)
            self.poses.append(pose)
            self.frames_id.append(frame_id)
            if feat is not None:
                self.features.append(rearrange(feat, 'c h w -> h w c'))
            self.last_frame_id = frame_id

            # Normalize patch if necessary
            patch = patch / 255.0 if patch.dtype != np.float32 or patch.max() > 1.0 else patch
            self.patches.append(patch)

            # Add descriptor
            if descriptor is not None:
                self.descriptors_list.append(descriptor.flatten())
            else:
                center_descriptor = feat[..., feat.shape[1] // 2, feat.shape[2] // 2].flatten()
                self.descriptors_list.append(center_descriptor)

            self.set_alive()
            return True
        return False

    def get_features_tensor(self):
        """Return features as a tensor."""
        return torch.tensor(self.features, dtype=torch.float32)

    def get_patches_tensor(self):
        """Return patches as a tensor."""
        return torch.tensor(np.stack(self.patches), dtype=torch.float32)

    def enhance(self, track):
        """
        Enhance the current track by merging with another track.

        Args:
            track (Track): Another track to merge with.

        Returns:
            bool: True if enhancement is successful, False if tracks have overlapping frames.
        """
        if set(self.frames_id).intersection(track.frames_id):
            return False

        if max(self.frames_id) < min(track.frames_id):
            # Prepend the new track
            self.poses = track.poses + self.poses
            self.points = track.points + self.points
            self.patches = track.patches + self.patches
            self.features = track.features + self.features
            self.frames_id = track.frames_id + self.frames_id
        else:
            # Append the new track
            self.poses += track.poses
            self.points += track.points
            self.patches += track.patches
            self.features += track.features
            self.frames_id += track.frames_id

        # Update the world point as a weighted average
        weight = len(self.points) / (len(self.points) + len(track.points))
        self.point_w = self.point_w * weight + track.point_w * (1 - weight)
        return True

    def same_as(self, track, threshold=0.05):
        """
        Check if the current track is the same as another track based on world point proximity.

        Args:
            track (Track): Another track to compare.
            threshold (float): Proximity threshold.

        Returns:
            bool: True if tracks are similar, False otherwise.
        """
        if set(self.frames_id).intersection(track.frames_id):
            return False
        return np.linalg.norm(self.point_w - track.point_w) < threshold

    def _is_same_point(self, pt):
        """Check if the given point matches the last tracked point."""
        return int(pt[0]) == int(self.points[-1][0]) and int(pt[1]) == int(self.points[-1][1])

    @classmethod
    def from_dict(cls, data):
        """Create a Track instance from a dictionary."""
        track = cls(data['patch_size_half'], data['distortion'], data['_id'])
        track.points = data['points']
        track.poses = data['poses']
        track.point_w = data['point_w']
        track.frames_id = data['frames_id']
        track.features = data['features']
        track.patches = data['patches']
        track.last_frame_id = data['last_frame_id']
        track.near = data['near']
        track.far = data['far']
        track.xyz_min = data['xyz_min']
        track.xyz_max = data['xyz_max']
        track.descriptors_list = data['descriptors_list']
        track.track_descriptor = data['track_descriptor']
        track._is_dead = data['_is_dead']
        return track

    def to_dict(self):
        """Convert the Track instance to a dictionary."""
        return {
            'points': self.points,
            'poses': self.poses,
            'point_w': self.point_w,
            'patch_size_half': self.patch_size_half,
            '_is_dead': self._is_dead,
            'frames_id': self.frames_id,
            'features': self.features,
            'patches': self.patches,
            'last_frame_id': self.last_frame_id,
            'near': self.near,
            'far': self.far,
            'xyz_min': self.xyz_min,
            'xyz_max': self.xyz_max,
            'descriptors_list': self.descriptors_list,
            'track_descriptor': self.track_descriptor,
            'distortion': self.distortion,
            '_id': self._id
        }
