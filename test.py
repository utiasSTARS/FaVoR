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
from lib.utils_favor.geom_utils import pose_error, IterativePnP
from lib.utils_favor.log_utils import print_info, print_success
from lib.utils_favor.misc_utils import matcher_fast, seed_env, init_device, parse_args, create_dataloader, \
    create_tracker, load_model, redirect2log, model2channels, log_results
from lib.utils_favor.visualizer_utils import score_to_rgb, visualize_camera_poses
from lib.models.favor_model import FaVoRmodel


def test(model, tracker, cfg, visualization=False):
    scene = cfg.data.scene
    if not (scene in ['chess', 'pumpkin', 'fire', 'heads', 'office', 'redkitchen', 'stairs']):
        raise Exception(
            "Scene must be one among: 'chess', 'pumpkin', 'fire', 'heads', 'office', 'redkitchen', 'stairs'")

    bboxes = model.get_bboxes()

    svfr_matches_count = []
    svfr_perc_matches = []
    matches_per_iter = [0., 0., 0.]
    #######################
    svfr_estimates = [0., 0., 0.]
    #######################
    init_dist_errors = []
    init_angle_errors = []
    estimated_dist_errors = [[], [], []]
    estimated_angle_errors = [[], [], []]
    #######################

    match_threshold = cfg.data.match_threshold[cfg.data.net_model]
    reprojection_error = cfg.data.reprojection_error[cfg.data.net_model]

    scene_folder = os.path.join(cfg.data.datadir, scene)
    # load closest poses from dataset/densevlad/7-Scenes/*.txt
    net_vlad_poses = {}
    with open(os.path.join("./datasets", 'densevlad', '7-Scenes', f'{scene}_top10.txt'), 'r') as f:
        prev_seq = None
        for line in f:
            seq, train_seq = line.strip().split(' ')
            if seq == prev_seq:
                continue
            net_vlad_poses[seq] = train_seq
            prev_seq = seq

    ground_truth_train_path = os.path.join(cfg.data.datadir, "COLMAP_gt", scene + "_train.txt")
    traing_gt_poses = {}
    focal = 0
    count = 0
    with open(ground_truth_train_path, 'r') as file:
        for line in file:
            img_path, qw, qx, qy, qz, x, y, z, f = line.split()

            R = Rotation.from_quat([float(qx), float(qy), float(qz), float(qw)]).as_matrix()
            t = np.array([float(x), float(y), float(z)])
            pose_train = np.eye(4)
            pose_train[:3, :3] = R
            pose_train[:3, 3] = t
            pose_train = np.linalg.inv(pose_train)

            traing_gt_poses[img_path] = pose_train

            focal += float(f)
            count += 1

    # get the average focal length
    focal /= count

    # create the camera matrix, principal point at the center
    K = np.array([
        [focal, 0., 320.],
        [0., focal, 240.],
        [0., 0., 1.]
    ])

    ground_truth_test_path = os.path.join(cfg.data.datadir, "COLMAP_gt", scene + "_test.txt")

    if visualization:
        cv2.namedWindow('Matches difference', cv2.WINDOW_NORMAL)
        cv2.namedWindow('SVFR Matches', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Network Matches', cv2.WINDOW_NORMAL)

    model.change_time_eval()
    total_voxels = len(model.voxels)
    print_info(f"Total voxels: {total_voxels}")
    gc.collect()

    outliers_estimates = []
    count_tests = 0
    with open(ground_truth_test_path, 'r') as file:
        lines = file.readlines()
        lines = sorted(lines)
        for line in tqdm(lines):
            count_tests += 1
            img_path, qw, qx, qy, qz, x, y, z, f = line.split()
            img_full_path = os.path.join(scene_folder, img_path)

            image = cv2.imread(img_full_path)
            image_height, image_width = image.shape[:2]

            pose = traing_gt_poses[net_vlad_poses[img_path]]

            R = Rotation.from_quat([float(qx), float(qy), float(qz), float(qw)]).as_matrix()
            t = np.array([float(x), float(y), float(z)])
            pose_gt = np.eye(4)
            pose_gt[:3, :3] = R
            pose_gt[:3, 3] = t
            pose_gt = np.linalg.inv(pose_gt)

            # compute perturbation error:
            err_angle_init, err_dist_init = pose_error(pose, copy.deepcopy(pose_gt))
            init_dist_errors.append(err_dist_init)
            init_angle_errors.append(err_angle_init)

            match_img = []
            # set the image for visualization
            if visualization:
                # white image 2 times the size of img
                match_img = np.ones((image_height, 2 * image_width, 3), dtype=np.uint8) * 255
                match_img[:, image_width:, :] = copy.deepcopy(image)

                gt_keypoints, _ = model.project_points(K=K, c2w=pose_gt, h=image_height, w=image_width)

                match_img_ = match_img.copy()
                for gt_kps in gt_keypoints:
                    match_img_ = cv2.circle(match_img_, (int(image.shape[1] + gt_kps[0]), int(gt_kps[1])), 3,
                                            (0, 255, 0),
                                            -1)
                    match_img_ = cv2.circle(match_img_, (int(gt_kps[0]), int(gt_kps[1])), 2,
                                            (0, 255, 0),
                                            -1)

            matched_landmarks, target_kpts, estimated_kpts, scores, camera_poses_list, estimated_dist_errors, estimated_angle_errors, iterate = iteratePnP(
                cfg, model, image, match_img, tracker, K, pose_gt, pose, match_threshold, reprojection_error,
                matcher_fast,
                svfr_estimates, matches_per_iter, estimated_dist_errors, estimated_angle_errors, visualization)

            # store the matches count
            svfr_matches_count.append(len(matched_landmarks))

            svfr_perc = 100. * (len(matched_landmarks) / total_voxels)
            svfr_perc_matches.append(svfr_perc)

            if visualization:
                for kp in estimated_kpts:
                    match_img = cv2.circle(match_img, (int(kp[0]), int(kp[1])), 2, (0, 255, 255), -1)

                cv2.putText(match_img, f'Matched keypoints: {svfr_perc:.2f} %',
                            (image_width - 100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            .7, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(match_img, f'Iter: {iterate}', (image_width - 100, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            .7, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(match_img, f'Original angular error: {err_angle_init:.2f} deg', (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            .7, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(match_img, f'Original linear error: {err_dist_init:.3f} m', (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            .7, (255, 255, 0), 2, cv2.LINE_AA)

                if (not np.isinf(estimated_dist_errors[-1]).any()) and len(estimated_dist_errors[-1]) > 0:
                    cv2.putText(match_img, f'Angular error: {estimated_angle_errors[-1][-1]:.2f} deg', (20, 70),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                .7, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(match_img, f'Linear error: {estimated_dist_errors[-1][-1]:.3f} m', (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                .7, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(match_img, f'No estimate', (20, 70),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                .7, (0, 0, 255), 2, cv2.LINE_AA)

                for kp1, kp2, score in zip(estimated_kpts, target_kpts, scores):
                    match_img = cv2.circle(match_img, (int(kp1[0]), int(kp1[1])), 1, (255, 0, 0), -1)

                    target_x = int(kp2[0] + image.shape[1])
                    match_img = cv2.circle(match_img, (target_x, int(kp2[1])), 3, (255, 0, 0), -1)
                    color = score_to_rgb((1.0 - score) / 0.1)
                    match_img = cv2.line(match_img, (int(kp1[0]), int(kp1[1])),
                                         (target_x, int(kp2[1])), color, 1)

                cv2.imshow('SVFR Matches', match_img)
                cv2.imshow('Matches difference', match_img_)
                cv2.waitKey(0)

            if visualization:
                visualize_camera_poses(camera_poses_list, bboxes, matched_landmarks)

        print_info(f"----------------WITH CLOSEST POSE PERTURBED----------------")
        print('---------------GENERAL SETUP-------------')
        print(f"Reprojection error: {reprojection_error}")
        print(f"Feature matching threshold: {match_threshold}")
        print(f"Voxel-grid size: {cfg.model_and_render.num_voxels} voxels")
        print('-----------------------------------------')
        print(f"Median initial distance error: {np.median(init_dist_errors):0.4} m")
        print(f"Median initial angle error: {np.median(init_angle_errors):0.4} deg")
        print(f"Max initial distance error: {np.max(init_dist_errors):0.4} m")
        print(f"Max initial angle error: {np.max(init_angle_errors):0.4} deg")
        print('-----------------------------------------')
        print(f"1st iter Median estimated distance error: {np.median(estimated_dist_errors[0]):0.4} m")
        print(f"1st iter Median estimated angle error: {np.median(estimated_angle_errors[0]):0.4} deg")
        print(f"1st iter Max estimated distance error: {np.max(estimated_dist_errors[0]):0.4} m")
        print(f"1st iter Max estimated angle error: {np.max(estimated_angle_errors[0]):0.4} deg")

        # critical estimates are the ones lower than 0.25 m and 2 degrees according to VRS-NeRF
        ests_low = np.sum(
            np.logical_and(np.array(estimated_dist_errors[0]) < 0.05, np.array(estimated_angle_errors[0]) < 5.))
        print(f"1st iter, <5 cm && <5deg: {ests_low}/{count_tests}")
        if svfr_estimates[1] > 0:
            print('-----------------------------------------')
            print(f"2nd iter Median estimated distance error: {np.median(estimated_dist_errors[1]):0.4} m")
            print(f"2nd iter Median estimated angle error: {np.median(estimated_angle_errors[1]):0.4} deg")
            print(f"2nd iter Max estimated distance error: {np.max(estimated_dist_errors[1]):0.4} m")
            print(f"2nd iter Max estimated angle error: {np.max(estimated_angle_errors[1]):0.4} deg")
            ests_low = np.sum(
                np.logical_and(np.array(estimated_dist_errors[1]) < 0.05, np.array(estimated_angle_errors[1]) < 5.))
            print(f"2nd iter, <5 cm && <5deg: {ests_low}/{count_tests}")
        if svfr_estimates[2] > 0:
            print('-----------------------------------------')
            print(f"3rd iter Median estimated distance error: {np.median(estimated_dist_errors[2]):0.4} m")
            print(f"3rd iter Median estimated angle error: {np.median(estimated_angle_errors[2]):0.4} deg")
            print(f"3rd iter Max estimated distance error: {np.max(estimated_dist_errors[2]):0.4} m")
            print(f"3rd iter Max estimated angle error: {np.max(estimated_angle_errors[2]):0.4} deg")
            ests_low = np.sum(
                np.logical_and(np.array(estimated_dist_errors[2]) < 0.05, np.array(estimated_angle_errors[2]) < 5.))
            print(f"3rd iter, <5 cm && <5deg: {ests_low}/{count_tests}")
        print('-----------------------------------------')
        print('Number of matches per iteration:')
        print(f"1st iter: {matches_per_iter[0] / svfr_estimates[0]:.2f}")
        if svfr_estimates[1] > 0:
            print(f"2nd iter: {matches_per_iter[1] / svfr_estimates[1]:.2f}")
        if svfr_estimates[2] > 0:
            print(f"3rd iter: {matches_per_iter[2] / svfr_estimates[2]:.2f}")
        print(f'Total number of iterations 1: {svfr_estimates[0]}')
        print(f'Total number of iterations 2: {svfr_estimates[1]}')
        print(f'Total number of iterations 3: {svfr_estimates[2]}')
        print('-----------------------------------------')
        print('Outliers estimates:', len(outliers_estimates))
        print('-----------------------------------------')
        print(f"Number of estimates: {len(estimated_angle_errors[2])}/{count_tests}")
        print('-----------------------------------------')
        print(f"SVFR matches/total points: {np.sum(svfr_matches_count)}/{total_voxels * count_tests}")
        print('-----------------------------------------')
        ###########################
        print(f"SVFR matches less than 10%: {np.sum(np.array(svfr_perc_matches) < 10)}")
        print('-----------------------------------------')
        ###########################
        print(f"SVFR with 0 matches: {np.sum(np.array(svfr_matches_count) == 0)}/{count_tests}")
        print('-----------------------------------------')
        ###########################
        print(f"SVFR less than 6 matches: {np.sum(np.array(svfr_matches_count) < 6)}/{count_tests}")

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
    f, original_stdout = redirect2log(cfg.root_dir)

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
    iter_pnp = IterativePnP(model=model,
                            K=dataloader.camera.K,
                            reprojection_error=cfg.data.reprojection_error[cfg.data.net_model],
                            match_threshold=cfg.data.match_threshold[cfg.data.net_model],
                            max_iter=tot_iterations,
                            tracker=tracker,
                            visualization=cfg.visualize)
    match_img = []
    starting_time_sec = time.time()
    for img, pose_gt, pose_prior in dataloader.get_test():
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
    log_results()


    # store the results:


    sys.stdout = original_stdout
    print_success("Testing done!")
