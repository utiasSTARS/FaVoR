import copy
import os
import sys
import time

import cv2

from lib.utils_favor.file_utils import store_obj
from lib.utils_favor.geom_utils import pose_error, IterativePnP
from lib.utils_favor.log_utils import print_info, print_success
from lib.utils_favor.misc_utils import seed_env, init_device, parse_args, create_dataloader, \
    create_tracker, load_model, redirect2log, model2channels, log_results, load_model_hf
from lib.models.favor_model import FaVoRmodel

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
    model = load_model_hf(cfg.root_dir, FaVoRmodel, cfg.data.dataset_type, cfg.data.scene, cfg.data.net_model)

    if model is None:
        raise Exception("Model not loaded, train a model first!")

    model.top_n(cfg.data.n_of_voxels)

    # create a log file and redirect stdout there
    f, original_stdout = redirect2log(cfg.root_dir, "results")

    channels = model2channels(cfg.data.net_model)
    print_info(f"\nChannels: {channels}")

    # ------------------- Start Testing -------------------
    print_info(f"Testing {cfg.data.scene} scene")
    tot_iterations = 3
    iter_pnp = IterativePnP(model=model,
                            K=dataloader.camera.K,
                            reprojection_error=cfg.data.reprojection_error[cfg.data.net_model],
                            match_threshold=cfg.data.match_threshold[cfg.data.net_model],
                            max_iter=tot_iterations,
                            tracker=tracker,
                            visualization=cfg.visualize)
    # match_img = []
    init_dist_errors, init_angle_errors = [], []
    tot_iter = 0
    starting_time_sec = time.time()
    for img, pose_gt, pose_prior in dataloader.get_test():
        tot_iter += 1

        err_angle_init, err_dist_init = pose_error(pose_prior, copy.deepcopy(pose_gt))
        init_dist_errors.append(err_dist_init)
        init_angle_errors.append(err_angle_init)

        # Iterate PnP to find pose
        iter_pnp(img, pose_gt, pose_prior)

    time_in_seconds = int(time.time() - starting_time_sec)
    # log results
    log_results(cfg, tot_iter, init_dist_errors, init_angle_errors, iter_pnp.estimated_dist_errors,
                iter_pnp.estimated_angle_errors, iter_pnp.matches_per_iter, model.get_n_voxels(), dense_vlad=True)

    # store the results:
    out_dir = os.path.join(cfg.basedir, cfg.expname)
    out_dir = os.path.join(out_dir, cfg.data.net_model)
    out_dir = os.path.join(out_dir, "results")

    print_info(f"Storing results in: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    store_obj(
        obj=[init_dist_errors, init_angle_errors, iter_pnp.estimated_dist_errors, iter_pnp.estimated_angle_errors,
             iter_pnp.matched_landmarks, iter_pnp.estimated_kpts, iter_pnp.favor_estimates, len(model.voxels)],
        path=os.path.join(out_dir, f"results_{int(time.time())}.obj"))

    sys.stdout = original_stdout
    print_success("Testing done!")
