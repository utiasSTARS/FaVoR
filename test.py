import copy
import gc
import os
import sys
import time

import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from lib.utils_favor.file_utils import store_obj
from lib.utils_favor.geom_utils import pose_error, IterativePnP, matcher_fast
from lib.utils_favor.log_utils import print_info, print_success
from lib.utils_favor.misc_utils import seed_env, init_device, parse_args, create_dataloader, \
    create_tracker, load_model, redirect2log, model2channels, log_results
from lib.utils_favor.visualizer_utils import score_to_rgb, visualize_camera_poses
from lib.models.favor_model import FaVoRmodel


def test(model, tracker, cfg, visualization=False):
    if visualization:
        cv2.destroyAllWindows()

    # store the errors
    out_dir = os.path.join(cfg.basedir, cfg.expname)
    out_dir = os.path.join(out_dir, cfg.data.net_model)
    out_dir = os.path.join(out_dir, "results")

    print_info(f"Storing results in: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    store_obj(
        obj=[init_dist_errors, init_angle_errors, estimated_dist_errors, estimated_angle_errors,
             outliers_estimates, matches_per_iter, svfr_estimates, count_tests, len(model.voxels)],
        path=os.path.join(out_dir, f"results_{time_in_seconds}.obj"))


if __name__ == '__main__':
    # to ensure reproducibility
    seed_env()

    # to ensure that the device is set correctly
    device = init_device()

    # load args
    cfg = parse_args()

    # ------------------- Define the Dataloader and Tracker -------------------
    dataloader = create_dataloader(dataset_type=cfg.data.dataset_type, data_path=cfg.data.datadir, scene=cfg.data.scene)

    tracker = create_tracker(net_model=cfg.net_model, K=dataloader.camera.K, patch_size_half=cfg.data.patch_size_half,
                             path=cfg.root_dir, distortion=dataloader.camera.distortion, log=False)

    # load the model
    model = load_model(cfg.root_dir, FaVoRmodel)

    if model is None:
        raise Exception("Model not loaded, train a model first!")

    top_voxels = 1500 if cfg.data.dataset_type.lower() == '7scenes' else 5000
    model.top_n(top_voxels)

    # create a log file and redirect stdout there
    f, original_stdout = redirect2log(cfg.root_dir, "test")

    # print options
    print_info(f"Reprojection error: {cfg.data.reprojection_error[cfg.data.net_model]}")
    print_info(f"Feature matching threshold: {cfg.data.match_threshold[cfg.data.net_model]}")
    print_info(f"Voxel-grid size: {cfg.model_and_render.num_voxels} voxels")
    print_info(f"Patch size half: {cfg.data.patch_size_half}")

    channels = model2channels(cfg.data.net_model)
    print_info(f"\nChannels: {channels}")

    # ------------------- Start Testing -------------------
    print_info(f"Testing {cfg.data.scene} scene")
    tot_iterations = 3
    cfg.visualize = False
    iter_pnp = IterativePnP(model=model,
                            K=dataloader.camera.K,
                            reprojection_error=cfg.data.reprojection_error[cfg.data.net_model],
                            match_threshold=cfg.data.match_threshold[cfg.data.net_model],
                            max_iter=tot_iterations,
                            tracker=tracker,
                            visualization=cfg.visualize)
    match_img = []
    init_dist_errors, init_angle_errors = [], []
    tot_iter = 0
    starting_time_sec = time.time()
    for img, pose_gt, pose_prior in dataloader.get_test():
        tot_iter += 1

        err_angle_init, err_dist_init = pose_error(pose_prior, copy.deepcopy(pose_gt))
        init_dist_errors.append(err_dist_init)
        init_angle_errors.append(err_angle_init)

        # set the image for visualization
        if cfg.visualize:
            # white image 2 times the size of img
            match_img = np.ones((dataloader.camera.height, 2 * dataloader.camera.width, 3), dtype=np.uint8) * 255
            match_img[:, dataloader.camera.width:, :] = copy.deepcopy(img)

            gt_keypoints, _ = model.project_points(K=dataloader.camera.K,
                                                   c2w=pose_gt,
                                                   h=dataloader.camera.height,
                                                   w=dataloader.camera.width)

            for gt_kps in gt_keypoints:
                match_img = cv2.circle(match_img, (int(img.shape[1] + gt_kps[0]), int(gt_kps[1])), 3,
                                       (0, 255, 0),
                                       -1)
                match_img = cv2.circle(match_img, (int(gt_kps[0]), int(gt_kps[1])), 2,
                                       (0, 255, 0),
                                       -1)

        # Iterate PnP to find pose
        iter_pnp(img, match_img, pose_gt, pose_prior)

    time_in_seconds = int(time.time() - starting_time_sec)
    # log results
    # cfg, init_dist_errors, init_angle_errors, estimated_dist_errors, estimated_angle_errors, svfr_estimates,
    #                 matches_per_iter, count_tests, dense_vlad=False
    log_results(cfg, tot_iter, init_dist_errors, init_angle_errors, iter_pnp.estimated_dist_errors,
                iter_pnp.estimated_angle_errors, iter_pnp.matches_per_iter, dense_vlad=True)

    # store the results:

    sys.stdout = original_stdout
    print_success("Testing done!")
