#  Copyright (c) 2023 Vincenzo Polizzi <vincenzo dot polizzi at mail dot utoronto dot ca>
#  (Space and Terrestrial Autonomous Robotic Systems Laboratory, University of Toronto,
#  Institute for Aerospace Studies).
#  This file is subject to the terms and conditions defined in the file
#  'LICENSE', which is part of this source code package.

import cv2
import numpy as np
import torch
from einops import rearrange

import thirdparty.third_party_loader

from lib.feature_extractors.ALIKE_extractor import AlikeExtractor
from lib.trackers.track import Track
from lib.trackers.base_tracker import BaseTracker
from lib.utils_favor.geom_utils import mnn_matcher, geometric_check, patch_creator
from lib.utils_favor.visualizer_utils import visualize_img, draw_kpts, visualize_images, to8b
from thirdparty.ALIKE.alike import configs as alike_configs
import copy
import gc


class AlikeTracker(BaseTracker):
    def __init__(self,
                 K: np.ndarray,
                 patch_size_half: int = 3,
                 max_points: int = 5000,
                 scores_th: float = 0.2,
                 top_k: int = -1,
                 model: str = "alike-n",
                 path: str = "",
                 log: bool = False,
                 min_track_length: int = 5,
                 distortion=None,
                 lkt=False):
        super().__init__(path, patch_size_half, min_track_length, distortion)
        self.active_tracks = []
        self.log = log
        self.K = K
        self.patch_size_half = patch_size_half
        self.net = AlikeExtractor(**alike_configs[model],
                                  device='cuda' if torch.cuda.is_available() else 'cpu',
                                  top_k=top_k,
                                  scores_th=scores_th,
                                  n_limit=max_points)
        self.lkt = lkt
        self.lk_params = dict(winSize=(10, 10),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def track(self, frame, pose):
        self.clean_tracks(min_len=self.min_track_length)

        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()

        # check if it is 8b
        if frame.dtype != np.uint8:
            frame = to8b(frame)

        if frame.shape[2] == 3:
            img_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2GRAY)

        img_rgb = frame

        # extract keypoints and descriptors
        res = self.net(img_rgb)
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
        def _track(kpts, desc, img_rgb, pose, feature_map):
            if len(self.prev_pts) == 0:
                self.prev_pts = kpts
                self.prev_frame = img_rgb
                self.prev_desc = desc

                for i in range(len(self.prev_pts)):
                    new_track = Track(patch_size_half=self.patch_size_half, distortion=self.distortion)
                    target = copy.deepcopy(patch_creator(feature_map, self.prev_pts[i], self.patch_size_half))
                    patch = copy.deepcopy(
                        patch_creator(self.prev_frame, self.prev_pts[i], self.patch_size_half, hwc=True))
                    new_track.append(self.prev_pts[i], pose, target, patch, self._frame_id)
                    self.tracks.append(new_track)
                if self.lkt:
                    self.prev_frame = img_gray
            else:
                if self.lkt:
                    kpts_lk, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, img_gray, self.prev_pts, None,
                                                                **self.lk_params)
                    mask = (st == 1).flatten()
                    kpts_match = kpts_lk[mask]
                    prev_pts_match = self.prev_pts[mask]
                else:
                    # match the descriptors
                    matches_all = mnn_matcher(desc, self.prev_desc, 0.7)
                    # update the points
                    kpts_match = kpts[matches_all[:, 0]]
                    prev_pts_match = self.prev_pts[matches_all[:, 1]]

                # ensure the points are inside the image
                mask_border = (kpts_match[:, 0] > 10) & (kpts_match[:, 0] < img_rgb.shape[1] - 10) & (
                            kpts_match[:, 1] > 10) & (
                                      kpts_match[:, 1] < img_rgb.shape[0] - 10)
                kpts_match = kpts_match[mask_border]
                prev_pts_match = prev_pts_match[mask_border]

                # filter the matches
                kpts_match, prev_pts_match, mask, _ = geometric_check(self.K, kpts_match, prev_pts_match, repr_error=2.)

                # check if the point is out:
                # for pt in kpts_match:
                #     if np.sum(img_rgb[int(pt[1]) - self.patch_size_half:int(pt[1]) + self.patch_size_half + 1,
                #               int(pt[0]) - self.patch_size_half:int(pt[0]) + self.patch_size_half + 1, :]) == 0:
                #         raise ValueError("Point is out of the colored image")

                # iterate over matches
                for mtc_id, (prev_pt, curr_pt) in enumerate(zip(prev_pts_match, kpts_match)):
                    target = copy.deepcopy(patch_creator(feature_map, curr_pt, self.patch_size_half))
                    patch = copy.deepcopy(patch_creator(img_rgb, curr_pt, self.patch_size_half, hwc=True))
                    added = False
                    for track_i in range(len(self.tracks)):
                        if self.tracks[track_i].get_last_frame_id() < self._frame_id - 1:
                            # no need to continue, the tracks are ordered
                            break

                        # if self.tracks[track_i].check(prev_pt):
                        if self.tracks[track_i].append(curr_pt, pose, target, patch, self._frame_id, prev_pt):
                            added = True
                            break

                    # if not added, init a new track
                    if not added:
                        new_track = Track(patch_size_half=self.patch_size_half, distortion=self.distortion)
                        new_track.append(curr_pt, pose, target, patch, self._frame_id)
                        self.tracks.append(new_track)

                # add the rest of the not matched points as new tracks:
                if not self.lkt:
                    for i in range(len(kpts)):
                        # not in matches_all is less than not in matches
                        if i not in matches_all[:, 0]:
                            target = copy.deepcopy(patch_creator(feature_map, kpts[i], self.patch_size_half))
                            patch = copy.deepcopy(patch_creator(img_rgb, kpts[i], self.patch_size_half, hwc=True))

                            new_track = Track(patch_size_half=self.patch_size_half, distortion=self.distortion)
                            new_track.append(kpts[i], pose, target, patch, self._frame_id)
                            self.tracks.append(copy.deepcopy(new_track))

                if self.log:
                    cv2.imshow("Matches", draw_kpts(img_rgb, prev_pts_match, kpts_match))
                    cv2.waitKey(1)
                    # visualize_images([self.prev_frame, img_rgb, draw_kpts(img_rgb, prev_pts_match, kpts_match)], 1, 3)

                # update the frame
                self.prev_frame = img_gray if self.lkt else img_rgb
                self.prev_pts = kpts_match.copy() if self.lkt else kpts.copy()
                self.prev_desc = desc.copy()
                if self.lkt:
                    if len(self.prev_pts) != 0 and len(self.prev_pts) < 200:
                        # check if pts are the same as the matched pts
                        c = 0
                        for p in kpts:
                            if np.linalg.norm(self.prev_pts - p, axis=1).min() > 5:
                                # add the new points
                                new_track = Track(patch_size_half=self.patch_size_half, distortion=self.distortion)
                                target = copy.deepcopy(patch_creator(feature_map, p, self.patch_size_half))
                                patch = copy.deepcopy(patch_creator(img_rgb, p, self.patch_size_half, hwc=True))
                                new_track.append(p, pose, target, patch, self._frame_id)
                                self.tracks.append(copy.deepcopy(new_track))
                                self.prev_pts = np.concatenate((self.prev_pts, p.reshape(1, 2)))
                                c += 1

                    print(f"REDETECTED {c} POINTS")

        _track(kpts, desc, img_rgb, pose, feature_map)

        if self.lkt:
            if len(self.tracks) > 3000:
                self.tracks = sorted(self.tracks, key=lambda x: x.get_last_frame_id(), reverse=True)
                # get the previous last frame id
                last_frame_id = self.tracks[0].get_last_frame_id()
                for i in range(len(self.tracks)):
                    if self.tracks[i].get_last_frame_id() != last_frame_id:
                        self.tracks[i:] = sorted(self.tracks[i:], key=lambda x: len(x), reverse=True)
                        self.tracks = self.tracks[:3000]
                        break

        # order tracks for last frame_id
        self.tracks = sorted(self.tracks, key=lambda x: x.get_last_frame_id(), reverse=True)
        gc.collect()

    def match(self, img1_, img2_):
        img1 = cv2.cvtColor(to8b(img1_.detach().cpu().numpy()), cv2.COLOR_BGR2RGB)
        res = self.net.run(img1)
        kpts = res['keypoints']
        desc = res['descriptors']

        img2 = cv2.cvtColor(to8b(img2_.detach().cpu().numpy()), cv2.COLOR_BGR2RGB)
        res2 = self.net.run(img2)
        kpts2 = res2['keypoints']
        desc2 = res2['descriptors']

        matches_all = mnn_matcher(desc, desc2)

        # update the points
        kpts_match = kpts[matches_all[:, 0]]
        prev_pts_match = kpts2[matches_all[:, 1]]

        # filter the matches
        kpts_match, prev_pts_match, mask, _ = geometric_check(self.K, kpts_match, prev_pts_match, repr_error=5.)

        # visualize the matches
        img_match = draw_kpts(img1, prev_pts_match, kpts_match)
        # write n of matches
        percentage_match = 100. * len(kpts_match) / len(kpts)
        cv2.putText(img_match, f"Matches: {percentage_match:.2f} %", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2,
                    cv2.LINE_AA)
        return img_match, len(kpts_match), percentage_match, len(kpts)

    def features_extractor(self, frame):
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()

        # check if it is 8b
        if frame.dtype != np.uint8:
            frame = to8b(frame)

        # the images are read using opencv, which is BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res = self.net(frame)
        kpts = res['keypoints']
        desc = res['descriptors']

        # store the score and feature maps
        score_map = rearrange(res['scores_map'], 'c h w -> h w c')
        feature_map = rearrange(res['feature_map'], 'c h w -> h w c')

        return kpts, desc, feature_map, score_map

    def descriptor_extractor(self, feature_map, kps):
        # extract descriptors
        with torch.no_grad():
            descriptors, offsets = self.net.desc_head(feature_map, kps)
            return descriptors, kps

    def keypoints_extractor(self, img, kps):
        # extract descriptors
        feature_map, score_map = self.net.extract_dense_map(img)
        descriptors, offsets = self.net.desc_head(feature_map, kps)
        return descriptors, kps

    def match_kps(self, desc, img2, match_thr):
        res2 = self.net(img2)
        kpts2 = res2['keypoints']
        desc2 = res2['descriptors']

        matches_all = mnn_matcher(desc, desc2, match_thr)

        # update the points
        target_kpts_match = kpts2[matches_all[:, 1]]

        return target_kpts_match, matches_all[:, 0]
