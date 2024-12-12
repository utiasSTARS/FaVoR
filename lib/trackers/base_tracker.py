#  Copyright (c) 2023 Vincenzo Polizzi <vincenzo.polizzi at mail dot utoronto dot ca>
#  (Space and Terrestrial Autonomous Robotic Systems Laboratory, University of Toronto,
#  Institute for Aerospace Studies).
#  This file is subject to the terms and conditions defined in the file
#  'LICENSE', which is part of this source code package.
import copy
import gc
import os
from copy import deepcopy

import cv2
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from lib.trackers.frame import Frame
from lib.trackers.track import Track
from lib.utils_favor.log_utils import print_warning, print_info
from lib.utils_favor.geom_utils import patch_creator, triangulate_point
from lib.utils_favor.file_utils import load_obj, store_obj
from lib.utils_favor.visualizer_utils import to8b

"""
BaseTracker class for feature tracking and storing.
===============================================================================
This class is designed to handle the tracking of features across frames,
manage track data, and apply specific configurations like patch size and
distortion correction. Tracks are loaded from and saved to a specified path.
"""


class BaseTracker:
    """
    Base class for feature tracking and storing.

    This class is designed to handle the tracking of features across frames,
    manage track data, and apply specific configurations like patch size and
    distortion correction. Tracks are loaded from and saved to a specified path.

    Attributes:
        path (str): The directory path where tracking data is stored or loaded from.
        prev_pts (list): A list of points from the previous frame.
        prev_frame (list): Data of the previous frame.
        prev_desc (list): Descriptors of features from the previous frame.
        tracks (list): A list of tracked feature data, loaded from a pickle file.
        tracks_ids (list): A list of unique IDs for each track.
        _frame_id (int): An internal counter to keep track of the current frame.
        patch_size_half (int): Half the size of the patch used for feature tracking.
        min_track_length (int): The minimum length required for a track to be valid.
        distortion: Parameters for distortion correction of feature points, if applicable.
    """

    def __init__(
            self,
            path: str = "",
            patch_size_half: int = 3,
            min_track_length: int = 5,
            distortion=None
    ):
        """
        Initializes the BaseTracker class.

        Args:
            path (str): Path to store or load tracker data. Default is an empty string.
            patch_size_half (int): Half the patch size for feature tracking. Default is 3.
            min_track_length (int): Minimum length of a track to be considered valid. Default is 5.
            distortion: Distortion parameters for undistorting points. Default is None.
        """
        self.path = path
        self.prev_pts = []  # Previous points
        self.prev_frame = []  # Previous frame data
        self.prev_desc = []  # Previous descriptors
        self.tracks = load_obj(os.path.join(self.path, "tracks.pkl"), Track, "Tracks ")  # Load tracks
        self.tracks_ids = []  # IDs of tracks
        self._frame_id = 0  # Frame counter
        # self._active_track_ids = []  # (Optional) Active track IDs, currently unused
        self.patch_size_half = patch_size_half
        self.min_track_length = min_track_length
        self.distortion = distortion  # Distortion parameters (if any)

    def __len__(self) -> int:
        """
        Get the number of tracks.

        Returns:
            int: The total number of tracks being managed by the tracker.
        """
        return len(self.tracks)

    def empty(self) -> bool:
        """
        Check if the tracker is empty.

        Returns:
            bool: True if no tracks are available, False otherwise.
        """
        return len(self) == 0

    def track_frame(self, frame, pose) -> 'Frame':
        """
        Run the tracker on a single frame.

        This method is meant to be implemented by subclasses to define
        how the tracking is performed on a given frame.

        Args:
            frame: The input frame to track features on.
            pose: The pose associated with the frame.

        Returns:
            Frame: A processed frame with tracked features (implementation-dependent).

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        raise NotImplementedError

    def features_extractor(self, img):
        """
        Extract features from an input image.

        This method is a placeholder for feature extraction logic, to be implemented by subclasses.

        Args:
            img: The input image from which features are to be extracted.

        Returns:
            Implementation-dependent feature descriptors.

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        raise NotImplementedError

    def match(self, img1_, img2_):
        """
        Match features between two images.

        This method defines the logic for finding correspondences between features
        in two images. It must be implemented by subclasses.

        Args:
            img1_: The first image.
            img2_: The second image.

        Returns:
            Implementation-dependent matching results.

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        raise NotImplementedError

    def run(self, imgs, poses):
        """
        Run the tracker on a sequence of images and their corresponding poses.

        This method processes a list of images and their associated poses,
        updating the tracker for each image-pose pair.

        Args:
            imgs (list): A list of images to track.
            poses (list): A list of poses corresponding to the images.

        Notes:
            The tracker will process each image-pose pair in sequence.
        """
        for img, pose in tqdm(zip(imgs, poses), total=len(imgs)):
            self.track(img, pose)
            self._frame_id += 1

    def run_once(self, img, pose):
        """
        Run the tracker on a single image and its corresponding pose.

        This method is used when tracking needs to be done on a single image-pose pair.

        Args:
            img: The image to track.
            pose: The pose corresponding to the image.

        Notes:
            The tracker will process the image and update the state based on the given pose.
        """
        self.track(img, pose)
        self._frame_id += 1

    def couple_features(self, h, w, count):
        """
        Couple the features with the tracks, used for large descriptor sizes.

        This method processes feature maps and associates them with the tracks.
        It is particularly useful for handling large descriptor sizes.

        Args:
            h (int): The height of the image.
            w (int): The width of the image.
            count (int): The number of training frames.

        Notes:
            The method assumes that feature maps are stored as `.npy` files in the "feature_maps" directory.
        """
        print_info("Coupling features with tracks...")
        map_tracks = {}

        # Create a mapping of track IDs to their index in the track list
        for i, t in enumerate(self.tracks):
            map_tracks[t.get_id()] = i

        # Generate frames for the given count
        frames = self.create_frames(list(range(count)))

        # Iterate over each frame
        for frame in tqdm(frames):
            # Load the feature map for the current frame
            feature_map_path = os.path.join(self.path, "feature_maps", f"feature_map_{frame.get_id()}.npy")
            feature_map = np.load(feature_map_path)

            # Convert the feature map to a torch tensor and resize it
            feature_map = torch.tensor(feature_map)
            feature_map = torch.nn.functional.interpolate(feature_map[None, ...], (h, w), mode='bilinear')[
                0].cpu().numpy()

            # Rearrange the dimensions of the feature map
            feature_map = rearrange(feature_map, 'c h w -> h w c')

            # Get the track IDs for the current frame
            t_ids = frame.get_v_ids()

            # Associate features with the tracks
            for t_id in t_ids:
                # Get the corresponding track from the map
                track = self.tracks[map_tracks[t_id]]
                assert track.get_id() == t_id, "Track id mismatch"

                # Get the point corresponding to the track in the current frame
                pt = track.points[track.frames_id.index(frame.get_id())]

                # Extract the feature patch for the point
                feat = copy.deepcopy(patch_creator(feature_map, pt, self.patch_size_half, True))

                # Ensure the feature size is correct
                if feat.shape[0] != 2 * self.patch_size_half + 1 or feat.shape[1] != 2 * self.patch_size_half + 1:
                    raise ValueError(f"Feature map size mismatch: {feat.shape}")

                # Append the feature to the track's feature list
                track.features.append(deepcopy(feat))
                del feat

            # Clean up feature map
            del feature_map

    def track(self, frame, pose):
        """
        Perform tracking on the frame and store poses and patches into tracks.

        This method is intended to be implemented by subclasses to handle
        specific tracking logic. It should process the frame and pose, then
        store the tracking information (such as poses and patches) into
        the tracks.

        Args:
            frame: The frame to track (usually an image or video frame).
            pose: The pose of the frame (position and orientation of the camera).

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        raise NotImplementedError

    def store_tracks(self):
        """
        Store the tracks in a pickle file.

        This method saves the current tracking information (stored in the
        `self.tracks` attribute) to a pickle file. The tracks are serialized
        and saved at the specified path.

        Notes:
            The file is saved as "tracks.pkl" in the directory specified by `self.path`.
        """
        store_obj(self.tracks, os.path.join(self.path, "tracks.pkl"))

    def store_patches(self):
        """
        Store the patches in a folder, used for large descriptor sizes.

        This method iterates over all the tracks, and for each track, it stores
        its associated patches in individual directories. The patches are saved
        as PNG files in a folder structure where each track has its own subfolder.
        The patch images are scaled by 255 and saved in the specified path.

        Notes:
            - The patches are stored in a folder named 'patches' inside the
              directory specified by `self.path`.
            - Each track gets its own subfolder, named by the track's ID.
            - Each patch is stored with the file name format:
              `{track_id}_{frame_id}.png`, where `track_id` is the ID of the
              track and `frame_id` is the ID of the frame the patch belongs to.
        """
        # Create the 'patches' directory if it does not exist
        patches_path = os.path.join(self.path, "patches")
        os.makedirs(patches_path, exist_ok=True)

        # Iterate over each track and store its patches
        for track in self.tracks:
            patch_path = os.path.join(patches_path, str(track.get_id()))
            os.makedirs(patch_path, exist_ok=True)

            # Iterate over the patches of the track and save them as images
            for i, p in enumerate(track.get_patches()):
                patch_filename = os.path.join(patch_path, f"{track.get_id()}_{track.frames_id[i]}.png")
                cv2.imwrite(patch_filename, (p * 255.0).astype(np.uint8))

    def store_images(self, imgs, path):
        """
        Store images in a specified folder.

        This method saves images corresponding to each track at the specified
        folder path. The images are stored in a subfolder named 'images'. For
        each track, it saves the images at the frames corresponding to the
        track’s frame IDs.

        Args:
            imgs: A list of images, where each image corresponds to a frame.
            path: The directory where the images will be stored.

        Notes:
            - The images are stored in a folder named 'images' within the
              provided `path`.
            - Each image is saved with the file name format:
              `{track_id}_{frame_id}.png`, where `track_id` is the ID of the
              track, and `frame_id` is the ID of the frame.
            - The pixel values of the images are scaled by 255 before being saved
              to ensure correct image intensity representation.
        """
        # Create the 'images' directory if it does not exist
        patches_path = os.path.join(path, "images")
        os.makedirs(patches_path, exist_ok=True)

        # Iterate over each track and save its corresponding images
        for track in self.tracks:
            for i, p in enumerate(track.get_patches()):
                # Get the image corresponding to the current frame ID
                img = imgs[track.frames_id[i]]

                # Save the image with the specified filename format
                img_filename = os.path.join(patches_path, f"{track.get_id()}_{track.frames_id[i]}.png")
                cv2.imwrite(img_filename, (img * 255.0).astype(np.uint8))

    def get_tracks(self, min_len=1, sort=False) -> [Track]:
        """
        Get the tracks that are longer than the specified minimum length.

        This method filters and returns the tracks that have a length greater than
        or equal to the specified `min_len`. Optionally, the tracks can be sorted
        by their length in descending order.

        Args:
            min_len: The minimum length of tracks to be returned. Tracks shorter
                     than this length will be excluded.
            sort: If True, the returned tracks will be sorted by length in
                  descending order.

        Returns:
            List of Track objects that meet the length criteria.

        Raises:
            ValueError: If no tracks are found after filtering.

        Notes:
            - If `min_len` is smaller than the minimum track length set in the
              object (`self.min_track_length`), a warning is displayed, and
              `min_len` is adjusted accordingly.
            - Tracks shorter than `min_len` are removed before returning the list.
        """
        # Ensure min_len is not smaller than the minimum track length
        if min_len < self.min_track_length:
            print_warning(f"min_len is smaller than min_track_length, setting min_len to {self.min_track_length}")
            min_len = self.min_track_length

        # Remove tracks that are shorter than the specified min_len
        if min_len > 0:
            self.remove_short_tracks(min_len)

        # Raise an error if no tracks remain after filtering
        if len(self.tracks) == 0:
            raise ValueError("No tracks found")

        # Return the tracks, optionally sorted by length
        if sort:
            return sorted(self.tracks, key=lambda x: len(x), reverse=True)

        return self.tracks

    def visualize_tracks(self, imgs, min_len=1):
        """
        Visualize the tracked patches on the images.

        This method iterates through the tracks and visualizes them by drawing
        lines connecting consecutive points in each track. The tracks are displayed
        on the image corresponding to the last frame of each track.

        Args:
            imgs: List of images where tracks will be visualized.
            min_len: The minimum length of the tracks to be visualized. Tracks shorter
                     than this length will be excluded.
        """
        # Ensure min_len is not smaller than the minimum track length
        if min_len < self.min_track_length:
            print_warning(f"min_len is smaller than min_track_length, setting min_len to {self.min_track_length}")
            min_len = self.min_track_length

        # Visualize tracks
        for track in self.get_tracks(min_len):
            # Get the last frame's image and convert it to an 8-bit format for display
            last_frame_id = track.get_last_frame_id()
            img_c = to8b(imgs[last_frame_id].detach().cpu().numpy().copy())
            img_c = cv2.cvtColor(img_c, cv2.COLOR_RGB2BGR)

            # Draw tracks as lines between consecutive points
            for i in range(len(track) - 1):
                point = track.points[i]
                point_next = track.points[i + 1]
                cv2.line(img_c, (int(point[0]), int(point[1])),
                         (int(point_next[0]), int(point_next[1])), (0, 0, 255), 2)

            # Mark the last point of the track with a circle
            img_c = cv2.circle(img_c, (int(track.points[-1][0]), int(track.points[-1][1])), 3, (255, 0, 255), -1)

            # Show the image with visualized tracks
            cv2.imshow('tracks', img_c)
            cv2.waitKey(0)  # Wait for a key press to continue

        # Close all OpenCV windows after visualization
        cv2.destroyAllWindows()

    def visualize_frames(self, frames, imgs):
        """Visualize the tracked patches squares on an image

        Args:
            frames: the frames [Frame]
            imgs: the images
        """
        for frame in frames:
            img = to8b(imgs[frame.get_id()].detach().cpu().numpy().copy())
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # create a mask for cropping the image in the patches
            mask = np.zeros_like(img)
            for point in frame.points:
                mask = cv2.circle(mask, (int(point[0]), int(point[1])), 5, (255, 255, 255), -1)
            # show only the circle in the image using the mask
            img = cv2.bitwise_and(img, mask)
            cv2.imshow('Patches', img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    def visualize_tracked_patches_per_frame(self, frames, size=(10, 10), imgs=None):
        """Visualize the tracked patches on the images

        Args:
            frames: the frames [Frame]
            size: the size of the patches
            imgs: the images
        """
        for frame in frames:
            # create a mask for cropping the image in the patches
            mask = np.zeros(size)
            for point, patch in zip(frame.points, frame.patches):
                p_x, p_y = int(point[0]), int(point[1])
                patch_size_half = int(patch.shape[0] / 2)
                mask[p_y - patch_size_half:p_y + patch_size_half + 1,
                p_x - patch_size_half:p_x + patch_size_half + 1] = patch

            if imgs is not None:
                img = to8b(imgs[frame.get_id()].detach().cpu().numpy().copy())
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) / 255.0
                img -= mask
                # normalize
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                cv2.imshow('img-patches', img)
            cv2.imshow('Patches', mask)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    def store_tracked_patches_per_frame(self, path, frames, size=(10, 10)):
        """Store the patches as an image file

        Args:
            path: the path where to store the patches
            frames: the frames [Frame]
            size: the size of the patches
        """
        for frame in frames:
            # create a mask for cropping the image in the patches
            mask = np.zeros((size[0], size[1], 4))
            for point, patch in zip(frame.points, frame.patches):
                p_x, p_y = int(point[0]), int(point[1])
                patch_size_half = int(patch.shape[0] / 2)
                # add 4th channel to the patch
                patch = np.concatenate((patch, np.ones_like(patch[:, :, :1])), axis=2)
                mask[p_y - patch_size_half:p_y + patch_size_half + 1,
                p_x - patch_size_half:p_x + patch_size_half + 1] = patch

            cv2.imshow('Patches', mask)
            cv2.waitKey(1)
            cv2.imwrite(os.path.join(path, f"r_{frame.get_id()}.png"), mask * 255.0)
        cv2.destroyAllWindows()

    def clean_tracks(self, min_len=5):
        """Remove the tracks older than 2 frames and shorter than min_len

        Args:
            min_len: the minimum length of the tracks
        """
        # l_0 = len(self.tracks)
        for track in self.tracks:
            if track.get_last_frame_id() < self._frame_id - 2 and len(track) < min_len:
                track.kill()
            if len(track) >= min_len:
                track.set_alive()
        self.tracks = [x for x in self.tracks if not x.is_dead()]

    def remove_short_tracks(self, min_len=5):
        """Remove the tracks that are shorter than min_len

        Args:
            min_len: the minimum length of the tracks
        """
        l_0 = len(self.tracks)
        for track in self.tracks:
            if len(track) < min_len:
                track.kill()
                # self._active_track_ids = np.delete(self._active_track_ids,
                #                                    np.where(self._active_track_ids == track.get_id()))
            else:
                track.set_alive()
        self.tracks = [x for x in self.tracks if not x.is_dead()]
        print_info(f"Cleaned {l_0 - len(self.tracks)} tracks")

    def triangulate_tracks(self, K, h, w, min_len=10):
        """ Compute all the information needed to create a voxel grid

        Args:
            K: the camera matrix
            h: the height of the image
            w: the width of the image
            min_len: the minimum length of the tracks
        """
        print_info("Triangulating points...")
        for track in tqdm(self.tracks):
            if len(track) < min_len:
                track.kill()
                continue
            poses = track.poses
            # triangulate the landmark using the observations
            point, refined_points, s, near, far = triangulate_point(poses, K, track.points,
                                                                    self.patch_size_half)
            if not isinstance(refined_points, list):
                raise Exception("Points are not list")

            if point == [] or refined_points == []:
                track.kill()
                continue

            # kill the track if refined_points are out the image
            if np.any(np.array(refined_points) < 5) or np.any(np.array(refined_points) > np.array([w, h]) - 5):
                track.kill()
                continue

            # check how far the refined points are from the tracked ones
            # if np.max(abs(np.array(refined_points) - np.array(track.points))) > 1.5:  # 2 px
            #     # remove the track, or update the patches
            #     track.kill()
            #     continue

            # update the track
            track.set_xyz_min(point - s)
            track.set_xyz_max(point + s)
            track.set_far(far)
            track.set_near(near)
            track.set_refined_points(refined_points)
            track.set_w_point(point)

        self.tracks = [x for x in self.tracks if not x.is_dead()]
        print_info(f"Total tracks: {len(self.tracks)}")

    def prune_tracks(self):
        """ Remove tracks of landmarks that are too close each other in 3D """
        print_warning("Pruning tracks...")
        # print_info(f"Initial tracks: {len(self.tracks)}")
        l_0 = len(self.tracks)
        self.tracks = [x for x in self.tracks if not x.is_dead()]
        gc.collect()
        print(f"Cleaned {l_0 - len(self.tracks)} tracks")

    def prune_tracks_fast(self):
        """ Remove tracks that are too close each other in 3D """
        raise DeprecationWarning("This function is deprecated")
        print_warning("Pruning tracks...")
        alive_tracks = [track for track in self.tracks if not track.is_dead()]
        l_0 = len(alive_tracks)

        for i, track in enumerate(alive_tracks):
            if track.is_dead():
                continue

            for j in range(i + 1, len(alive_tracks)):
                track_2 = alive_tracks[j]
                if track_2.is_dead():
                    continue

                if track.same_as(track_2, 0.01):
                    track.enhance(track_2)
                    track_2.kill()
                elif track.same_as(track_2, 0.05):
                    if len(track) > len(track_2):
                        track_2.kill()
                    else:
                        track.kill()
                        break

        self.tracks = [x for x in alive_tracks if not x.is_dead()]
        print(f"Cleaned {l_0 - len(self.tracks)} tracks")

    def create_frames(self, frames_id) -> [Frame]:
        """Create a frame object from the tracks

        Args:
            frames_id: the frames id
        Returns:
            List of frames
        """
        frames = []
        for f_id in frames_id:
            poses, points, patches, features, v_ids = [], [], [], [], []
            for t in self.tracks:
                if f_id in t.get_frames_ids():
                    pose, point, patch, feature = t.get_at_frame(f_id)
                    if pose is None:
                        continue
                    poses, points, patches, features, v_ids = poses + [pose], points + [point], patches + [
                        patch], features + [feature], v_ids + [t.get_id()]
            if len(points) == 0:
                continue
            f = Frame(f_id, poses, points, patches, features, v_ids)
            frames.append(f)
        return frames

    def merge_tracks(self):
        print_info(f"Merging tracks...")
        print_info(f"Initial tracks: {len(self.tracks)}")
        tracks_descriptor = []
        for track in self.tracks:
            track.init_descriptor()
            tracks_descriptor.append(track.track_descriptor)
        tracks_descriptor = np.array(tracks_descriptor)
        sim = tracks_descriptor @ tracks_descriptor.T
        sim = sim > 0.9
        sim = np.triu(sim, 1)
        sim = np.where(sim)
        for i, j in zip(sim[0], sim[1]):
            if not (self.tracks[i].is_dead() or self.tracks[j].is_dead()):
                self.tracks[i].enhance(self.tracks[j])
                self.tracks[j].kill()

        # remove the dead tracks
        self.prune_tracks()
        print_info(f"Final tracks: {len(self.tracks)}")

    def triangulate(self, K, H, W, min_track_length=10):
        """ Perform Structure from Motion on the tracks """
        print_info(f"Reducing tracks to max 25000...")
        # order the tracks by length
        self.tracks = sorted(self.tracks, key=lambda x: len(x), reverse=True)
        if len(self.tracks) > 25000:
            self.tracks = self.tracks[:25000]
        print_info(f"Total tracks: {len(self.tracks)}")
        print_info(f"Merging tracks...")
        self.merge_tracks()
        # keep the tracks that are longer than min_track_length
        self.tracks = [track for track in self.tracks if len(track) > min_track_length]

        print_info(f"Start tracks traingulation...")
        print_info(f"Total tracks: {len(self.tracks)}")
        self.triangulate_tracks(K, H, W, min_track_length)
