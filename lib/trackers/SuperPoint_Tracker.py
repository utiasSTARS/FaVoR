#  Copyright (c) 2023 Vincenzo Polizzi <vincenzo dot polizzi at mail dot utoronto dot ca>
#  (Space and Terrestrial Autonomous Robotic Systems Laboratory, University of Toronto,
#  Institute for Aerospace Studies).
#  This file is subject to the terms and conditions defined in the file
#  'LICENSE', which is part of this source code package.

import copy
import gc
import os.path

import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor
from overrides import override

from lib.feature_extractors.SuperPoint_extractor import SuperPointExtractor
from lib.trackers.track import Track
from lib.trackers.base_tracker import BaseTracker
from lib.utils_favor.geom_utils import mnn_matcher, geometric_check, patch_creator
from lib.utils_favor.visualizer_utils import draw_kpts, visualize_images
from copy import deepcopy


class SuperPointTracker(BaseTracker):
    def __init__(self,
                 K: np.ndarray,
                 patch_size_half: int = 3,
                 max_points: int = 1000,
                 scores_th: float = 0.2,
                 top_k: int = -1,
                 model: str = "superpoint_v6_from_tf",
                 path: str = "",
                 log: bool = False,
                 min_track_length: int = 5,
                 distortion=None):
        """
        Initialize the SuperPointTracker.

        Args:
            K (np.ndarray): Camera intrinsic matrix.
            patch_size_half (int): Half size of the patch.
            max_points (int): Maximum number of points.
            scores_th (float): Score threshold.
            top_k (int): Top K keypoints to consider.
            model (str): Model name.
            path (str): Path to save logs.
            log (bool): Whether to log the process.
            min_track_length (int): Minimum track length.
            distortion: Distortion parameters.
        """
        super().__init__(path, patch_size_half, min_track_length, distortion)
        self.active_tracks = []
        self.log = log
        self.K = K
        self.patch_size_half = patch_size_half
        self.net = SuperPointExtractor(model_name=model,
                                       device='cuda' if torch.cuda.is_available() else 'cpu')

    @override
    def track(self, frame, pose):
        """
        Track keypoints in the given frame.

        Args:
            frame: The current frame.
            pose: The current pose.

        Returns:
            None
        """
        self.clean_tracks(min_len=self.min_track_length)

        # image to numpy, rgb
        if isinstance(frame, np.ndarray):
            img_rgb = torch.tensor(frame, dtype=torch.float32)

        # normalize the image
        img_rgb = img_rgb / 255.
        # extract keypoints and descriptors
        res = self.net(img_rgb.permute(2, 0, 1)[None, ...])
        kpts = res['keypoints']
        desc = res['descriptors']

        # remove points too close to the border, 10 px margin
        mask_border = (kpts[:, 0] > 10) & (kpts[:, 0] < img_rgb.shape[1] - 10) & (kpts[:, 1] > 10) & (
                kpts[:, 1] < img_rgb.shape[0] - 10)
        kpts = kpts[mask_border]
        desc = desc[mask_border]

        # store the score and feature maps
        feature_map = res['feature_map']

        # track the points
        def _track(kpts, desc, img, pose, feature_map):
            """
            Internal function to track keypoints.

            Args:
                kpts: Keypoints.
                desc: Descriptors.
                img: RGB image.
                pose: Current pose.
                feature_map: Feature map.

            Returns:
                None
            """
            img_rgb = cv2.cvtColor(to8b(img.detach().cpu().numpy()), cv2.COLOR_BGR2RGB)
            if len(self.prev_pts) == 0:
                self.prev_pts = kpts
                self.prev_frame = img_rgb
                self.prev_desc = desc

                for i in range(len(self.prev_pts)):
                    new_track = Track(patch_size_half=self.patch_size_half, distortion=self.distortion)
                    target = None
                    patch = copy.deepcopy(
                        patch_creator(self.prev_frame, self.prev_pts[i], self.patch_size_half, hwc=True))
                    new_track.append(self.prev_pts[i], pose, target, patch, self._frame_id,
                                     descriptor=self.prev_desc[i])
                    self.tracks.append(new_track)
            else:
                # match the descriptors
                matches_all = mnn_matcher(desc, self.prev_desc, 0.8)
                # update the points
                kpts_match = kpts[matches_all[:, 0]]
                prev_pts_match = self.prev_pts[matches_all[:, 1]]
                # filter the matches
                kpts_match, prev_pts_match, mask, _ = geometric_check(self.K, kpts_match, prev_pts_match, repr_error=5.)

                # iterate over matches
                for mtc_id, (prev_pt, curr_pt) in enumerate(zip(prev_pts_match, kpts_match)):
                    target = None
                    patch = copy.deepcopy(patch_creator(img_rgb, curr_pt, self.patch_size_half, hwc=True))
                    added = False
                    for track_i in range(len(self.tracks)):
                        if self.tracks[track_i].get_last_frame_id() < self._frame_id - 1:
                            break

                        if self.tracks[track_i].append(curr_pt, pose, target, patch, self._frame_id, prev_pt,
                                                       descriptor=desc[matches_all[mtc_id, 0]]):
                            added = True
                            break

                    if not added:
                        new_track = Track(patch_size_half=self.patch_size_half, distortion=self.distortion)
                        new_track.append(curr_pt, pose, target, patch, self._frame_id,
                                         descriptor=desc[matches_all[mtc_id, 0]])
                        self.tracks.append(new_track)

                for i in range(len(kpts)):
                    if i not in matches_all[:, 0]:
                        target = None
                        patch = copy.deepcopy(patch_creator(img_rgb, kpts[i], self.patch_size_half, hwc=True))

                        new_track = Track(patch_size_half=self.patch_size_half, distortion=self.distortion)
                        new_track.append(kpts[i], pose, target, patch, self._frame_id, descriptor=desc[i])
                        self.tracks.append(deepcopy(new_track))

                if self.log:
                    visualize_images([self.prev_frame, img_rgb, draw_kpts(img_rgb, prev_pts_match, kpts_match)], 1, 3)

                self.prev_frame = img_rgb
                self.prev_pts = kpts.copy()
                self.prev_desc = desc.copy()

        _track(kpts, desc, img_rgb, pose, feature_map)
        path = os.path.join(self.path, "feature_maps")
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path, f"feature_map_{self._frame_id}.npy"), feature_map)

        self.tracks = sorted(self.tracks, key=lambda x: x.get_last_frame_id(), reverse=True)
        gc.collect()

    def features_extractor(self, img):
        """
        Extract features from the given image.

        Args:
            img: The input image.

        Returns:
            tuple: Keypoints, descriptors, feature map, score map.
        """
        if not isinstance(img, torch.Tensor):
            img_tensor = ToTensor()(img)
        else:
            img_tensor = img.permute(2, 0, 1)
        h, w = img_tensor.shape[1], img_tensor.shape[2]
        res = self.net(img_tensor[None, ...])
        kpts = res['keypoints']
        desc = res['descriptors']

        feature_map = torch.tensor(res['feature_map'], dtype=torch.float32)
        feature_map = torch.nn.functional.interpolate(feature_map[None, ...], (h, w), mode='bilinear')[
            0].permute(1, 2, 0)

        return kpts, desc, feature_map.cpu().numpy(), np.transpose(res['feature_map'], (1, 2, 0))

    def descriptor_extractor(self, feature_map, kps):
        """
        Extract descriptors from the feature map and keypoints.

        Args:
            feature_map: The feature map.
            kps: Keypoints.

        Returns:
            tuple: Descriptors, keypoints.
        """
        with torch.no_grad():
            descriptors, offsets = self.net.desc_head(feature_map, kps)
            return descriptors, kps

    def keypoints_extractor(self, img, kps):
        """
        Extract keypoints from the image.

        Args:
            img: The input image.
            kps: Keypoints.

        Returns:
            tuple: Descriptors, keypoints.
        """
        feature_map, score_map = self.net.extract_dense_map(img)
        descriptors, offsets = self.net.desc_head(feature_map, kps)
        return descriptors, kps

    def match(self, img1, img2):
        """
        Match keypoints between two images.

        Args:
            img1: First image.
            img2: Second image.

        Returns:
            tuple: Matched image, number of matches, percentage of matches, number of keypoints.
        """
        res = self.net(img1.permute(2, 0, 1)[None, ...])
        kpts = res['keypoints']
        desc = res['descriptors']

        res2 = self.net(img2.permute(2, 0, 1)[None, ...])
        kpts2 = res2['keypoints']
        desc2 = res2['descriptors']

        matches_all = mnn_matcher(desc, desc2, 0.8)

        kpts_match = kpts[matches_all[:, 0]]
        prev_pts_match = kpts2[matches_all[:, 1]]

        kpts_match, prev_pts_match, mask, _ = geometric_check(self.K, kpts_match, prev_pts_match, repr_error=2.)

        img_match = draw_kpts(img1, prev_pts_match, kpts_match)
        percentage_match = 100. * len(kpts_match) / len(kpts)
        cv2.putText(img_match, f"Matches: {percentage_match:.2f} %", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2,
                    cv2.LINE_AA)
        return img_match, len(kpts_match), percentage_match, len(kpts)

    def match_kps(self, desc, img2, match_thr):
        """
        Match keypoints between descriptors and an image.

        Args:
            desc: Descriptors.
            img2: The second image.
            match_thr: Matching threshold.

        Returns:
            tuple: Target keypoints match, matched indices.
        """
        if isinstance(img2, np.ndarray):
            img2 = torch.tensor(img2, dtype=torch.float32)

        img2 = img2 / 255.

        res2 = self.net(img2.permute(2, 0, 1)[None, ...])
        kpts2 = res2['keypoints']
        desc2 = res2['descriptors']

        matches_all = mnn_matcher(desc, desc2, match_thr)

        target_kpts_match = kpts2[matches_all[:, 1]]

        return target_kpts_match, matches_all[:, 0]
