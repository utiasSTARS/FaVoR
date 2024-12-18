import sys
import time

import cv2
import numpy as np
from tqdm import tqdm

from lib.utils_favor.log_utils import print_info, print_success
from lib.utils_favor.transform_utils import CV2O3D
from lib.utils_favor.visualizer_utils import visualize_camera_poses_and_points
from lib.utils_favor.misc_utils import seed_env, init_device, parse_args, create_dataloader, create_tracker, \
    redirect2log

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

    if tracker.empty():
        # start timing
        start_time = int(time.time())

        # create a log file and redirect stdout there
        f, original_stdout = redirect2log(cfg.root_dir, "track")

        print_info(f"Tracking {cfg.data.scene} scene")

        # run the tracker on the dataset
        counter = 0
        for img, cam_pose in dataloader.get_train():
            tracker.run_once(img, cam_pose)
            counter += 1

        # triangulate tracked points
        tracker.triangulate(dataloader.camera.K, dataloader.camera.height, dataloader.camera.width,
                            cfg.data.min_track_length)

        # if the features are large, save restore them from disk and associate with the points
        if len(tracker.tracks[0].features) == 0:
            tracker.tracks = sorted(tracker.tracks, key=lambda x: len(x), reverse=True)[:3000]
            tracker.couple_features(dataloader.camera.height, dataloader.camera.width, counter)

        # log time
        print_info(f"Tracking took {(time.time() - start_time)} seconds")

        # store the tracks
        tracker.store_tracks()

        # close the log file and reset stdout
        f.close()
        sys.stdout = original_stdout

    # visualize the tracks if flag visualise is set
    camera_poses_list = []
    if cfg.visualize:
        # load all the landmarks tracked
        landmarks = []
        for track in tracker.tracks:
            h_coords = np.concatenate((track.get_w_point(), [1.0]))
            landmarks.append(h_coords)

        landmarks = np.array(landmarks)

        print_info(f"Visualizing tracked points on {cfg.data.scene} scene test images")
        for img, cam_pose, _ in dataloader.get_test():
            camera_poses_list.append(CV2O3D(cam_pose))
            cam_pose = np.linalg.inv(cam_pose)

            # project the landmarks to the image
            mask = np.ones(len(landmarks), dtype=bool)

            bearing_vecs = np.dot(cam_pose[:3], landmarks.T)
            mask &= bearing_vecs[2] > 0

            pts = np.dot(dataloader.camera.K, bearing_vecs).T
            pts /= pts[:, 2].reshape(-1, 1)
            pts = pts[:, :2]

            mask &= (pts[:, 0] > 5) & (pts[:, 1] > 5) & (pts[:, 0] < dataloader.camera.width - 5) & (
                    pts[:, 1] < dataloader.camera.height - 5)
            pts = pts[mask]

            for pt in pts:
                cv2.circle(img, tuple(pt.astype(int)), 3, (0, 0, 255), -1)

            cv2.imshow("img", img)
            cv2.waitKey(1)

        cv2.destroyAllWindows()

        # visualize poses and landmarks
        landmarks = []
        for track in tqdm(tracker.tracks):
            if len(track) > 10:
                landmarks.append(track.get_w_point())

        visualize_camera_poses_and_points(camera_poses_list, landmarks)

print_success("Tracking completed successfully")
