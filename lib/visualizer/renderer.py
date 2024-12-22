#  Copyright (c) 2023 Vincenzo Polizzi <vincenzo dot polizzi at mail dot utoronto dot ca>
#  (Space and Terrestrial Autonomous Robotic Systems Laboratory, University of Toronto,
#  Institute for Aerospace Studies).
#  This file is subject to the terms and conditions defined in the file
#  'LICENSE', which is part of this source code package.

import cv2
import numpy as np

from lib.data_loaders.Dataloader import Dataloader
from lib.models.favor_model import FaVoRmodel
from lib.trackers.base_tracker import BaseTracker
from lib.utils_favor.geom_utils import IterativePnP


class FavorRender:

    def __init__(self, cfg, tracker: BaseTracker, model: FaVoRmodel, dataloader: Dataloader):
        self.pointer = 0
        self.o3d_view = None
        self.current_img = None

        self.current_gt_pose = None
        self.current_prior_pose = None

        self.dataloader = dataloader

        self.current_img, self.current_gt_pose, self.current_prior_pose = self.dataloader.get_test_line_at(self.pointer)

        tot_iterations = 3

        self.iter_pnp = IterativePnP(model=model,
                                     K=dataloader.camera.K,
                                     reprojection_error=cfg.data.reprojection_error[cfg.data.net_model],
                                     match_threshold=cfg.data.match_threshold[cfg.data.net_model],
                                     max_iter=tot_iterations,
                                     tracker=tracker,
                                     visualization=True)

        cv2.namedWindow('Matches', cv2.WINDOW_AUTOSIZE)

    def set_reprojection_error(self, reprojection_error):
        self.iter_pnp.reprojection_error = reprojection_error

    def set_match_threshold(self, match_threshold):
        self.iter_pnp.match_threshold = match_threshold

    def current_image(self):
        self.current_img, self.current_gt_pose, self.current_prior_pose = self.dataloader.get_test_line_at(self.pointer)
        self.update_image()

    def next_image(self):
        self.pointer += 1
        self.current_img, self.current_gt_pose, self.current_prior_pose = self.dataloader.get_test_line_at(self.pointer)
        self.update_image()
        return self.current_gt_pose, self.current_prior_pose

    def previous_image(self):
        self.pointer -= 1
        self.current_img, self.current_gt_pose, self.current_prior_pose = self.dataloader.get_test_line_at(self.pointer)
        self.update_image()
        return self.current_gt_pose, self.current_prior_pose

    def update_image(self):
        # resize o3d view to the same size as the current image
        cv2.imshow('Matches', self.current_img)
        cv2.waitKey(1)

    def localize_image(self):
        self.iter_pnp(self.current_img, self.current_gt_pose, self.current_prior_pose)
        # self.update_image()
        if self.iter_pnp.estimated_angle_errors[2] is not np.inf:
            return self.iter_pnp.camera_poses_list[-1][-1], self.iter_pnp.matched_landmarks[-1][-1]
        return None

    def stop(self):
        cv2.destroyAllWindows()
