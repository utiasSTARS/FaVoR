#  Copyright (c) 2023 Vincenzo Polizzi <vincenzo dot polizzi at mail dot utoronto dot ca>
#  (Space and Terrestrial Autonomous Robotic Systems Laboratory, University of Toronto,
#  Institute for Aerospace Studies).
#  This file is subject to the terms and conditions defined in the file
#  'LICENSE', which is part of this source code package.

import copy
import numpy as np
import cv2
import poselib as plib
from typing import Tuple, List
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

from lib.utils_favor.log_utils import print_warning
from lib.utils_favor.visualizer_utils import visualize_matches

"""
Collection of geometric utility functions.
"""


def matcher_fast(model, image, tracker, K, pose, match_thr) -> Tuple:
    """
    Perform fast matching between rendered descriptors and image keypoints.

    Args:
        model: The model used for rendering and matching.
        image (np.ndarray): Input image of shape (H, W).
        tracker: Object responsible for matching keypoints.
        K (np.ndarray): Intrinsic camera matrix.
        pose (np.ndarray): Camera pose (4x4 transformation matrix).
        match_thr (float): Matching threshold.

    Returns:
        tuple: Matched rendered keypoints, matched target keypoints, matched landmarks,
               all rendered points, rendered landmarks, and scores.
    """
    image_height, image_width = image.shape[:2]

    # Render visible voxels
    rendered_descs, rendered_pts, rendered_landmasks = model.point_render(
        K=K,
        p_wc=pose,
        h=image_height,
        w=image_width
    )

    # Match the rendered descriptors with the image
    matched_target_kpts, mask, scores = tracker.match_kps(
        rendered_descs,
        image,
        match_thr
    )

    # Get the matched rendered keypoints and landmarks
    matched_rendered_kpts = rendered_pts[mask]
    matched_landmarks = rendered_landmasks[mask]

    # Return empty lists if no matches are found
    if len(matched_rendered_kpts) == 0:
        return [], [], [], []

    return matched_rendered_kpts, matched_target_kpts, matched_landmarks, rendered_pts, rendered_landmasks, scores


def mnn_matcher(desc1: np.ndarray, desc2: np.ndarray, match_thr: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds matches between two descriptor lists (desc1 and desc2) using mutual nearest neighbor (MNN) matching.

    Args:
        desc1 (np.ndarray): Descriptor array from the first set of keypoints.
        desc2 (np.ndarray): Descriptor array from the second set of keypoints.
        match_thr (float): Threshold for similarity score (default: 0.9).

    Returns:
        np.ndarray: Array of matches with shape (N, 2), where each row is a pair of matched indices.
        np.ndarray: Array of similarity scores for each match.
    """
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return np.zeros((0, 2), dtype=int), np.zeros(0, dtype=float)

    # Compute similarity matrix
    sim = desc1 @ desc2.T
    sim[sim < match_thr] = 0

    # Nearest neighbors in both directions
    nn12 = np.argmax(sim, axis=1)
    nn21 = np.argmax(sim, axis=0)

    # Perform mutual check
    ids1 = np.arange(sim.shape[0])
    mask = (ids1 == nn21[nn12])
    matches = np.stack([ids1[mask], nn12[mask]], axis=1)

    scores = np.array(sim[mask, nn12[mask]])
    return matches, scores


def geometric_check(
        K: np.ndarray,
        pts1: np.ndarray,
        pts2: np.ndarray,
        repr_error: float
) -> Tuple[List[np.ndarray], List[np.ndarray], List[bool], np.ndarray]:
    """
    Performs an Essential matrix check on correspondences between pts1 and pts2.

    Args:
        K (np.ndarray): Camera intrinsic matrix.
        pts1 (np.ndarray): Points from the first image (Nx2).
        pts2 (np.ndarray): Points from the second image (Nx2).
        repr_error (float): Maximum allowed reprojection error for RANSAC.

    Returns:
        Tuple: A tuple containing:
            - Filtered points from pts1 (List[np.ndarray]).
            - Filtered points from pts2 (List[np.ndarray]).
            - Inlier mask as a list of booleans (List[bool]).
            - Transformation matrix T (np.ndarray) of shape (4, 4).
    """
    if len(pts1) != len(pts2):
        raise ValueError("pts1 and pts2 must have the same number of points.")
    if len(pts1) < 4:
        return [], [], [], np.eye(4, dtype=np.float32)

    # Convert points to the correct format
    pts1 = np.ascontiguousarray(pts1, dtype=np.float32)
    pts2 = np.ascontiguousarray(pts2, dtype=np.float32)

    # Compute Essential matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, repr_error, maxIters=1000)

    try:
        # Recover pose
        inlier_mask = mask.ravel().astype(bool)
        _, R, t, _ = cv2.recoverPose(E, pts1[inlier_mask], pts2[inlier_mask], K)

        # Build the transformation matrix
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()

    except cv2.error:
        return [], [], [], np.eye(4, dtype=np.float32)

    # Create matches mask
    matchesMask = mask.ravel().tolist()

    return pts1[inlier_mask].tolist(), pts2[inlier_mask].tolist(), matchesMask, T


def patch_creator(
        target_map: np.ndarray, pt: Tuple[float, float], patch: int, hwc: bool = False) -> np.ndarray:
    """
    Extracts a square patch around a given point from the target map.

    Args:
        target_map (np.ndarray): The map (image or feature map) to extract the patch from.
        pt (Tuple[int, int]): The (x, y) coordinates of the center point for the patch.
        patch (int): Half the width of the patch (patch size is 2 * patch + 1).
        hwc (bool): Whether the target map has (height, width, channels) order (default: False).

    Returns:
        np.ndarray: Extracted patch of size (2*patch+1, 2*patch+1, ...) or (..., 2*patch+1, 2*patch+1).
    """
    # Compute patch boundaries
    r_x_0, r_y_0 = pt[1] - patch, pt[0] - patch
    r_x_1, r_y_1 = pt[1] + patch + 1, pt[0] + patch + 1
    # cast to int
    r_x_0, r_y_0 = int(r_x_0), int(r_y_0)
    r_x_1, r_y_1 = int(r_x_1), int(r_y_1)

    # Extract patch based on format
    if hwc:
        return target_map[r_x_0:r_x_1, r_y_0:r_y_1, ...]

    return target_map[..., r_x_0:r_x_1, r_y_0:r_y_1]


def triangulate_point(
        T_wc_list: List[np.ndarray],
        K: np.ndarray,
        pts: List[Tuple[float, float]],
        patch_size_half: int = 4
) -> Tuple[np.ndarray, List[np.ndarray], float, float, float]:
    """
    Triangulates a 3D point from multiple camera poses and computes the bounding box size and location.

    Args:
        T_wc_list (List[np.ndarray]): List of 4x4 transformation matrices from world to camera coordinates.
        K (np.ndarray): Intrinsic matrix of the camera (3x3).
        pts (List[Tuple[float, float]]): List of 2D points in the image planes of the cameras.
        patch_size_half (int, optional): Half of the patch size for bounding box calculation. Default is 4.

    Returns:
        Tuple:
            - np.ndarray: Estimated 3D point in world coordinates (shape: 3,).
            - List[np.ndarray]: Refined 2D points projected onto all camera frames.
            - float: Bounding box size.
            - float: Dummy value (kept for compatibility).
            - float: Residual cost or error measure.
    """
    # Invert transformations to get camera-to-world matrices
    T_cw_list = [np.linalg.inv(T_wc) for T_wc in T_wc_list]

    # Normalize points using the inverse intrinsic matrix
    pts_norm = np.zeros((len(T_cw_list), 2))
    K_inv = np.linalg.inv(K)
    for i in range(len(T_cw_list)):
        pts_norm[i] = (K_inv @ np.array([pts[i][0], pts[i][1], 1.0]))[:2]

    # Use the first camera as the reference
    T_cw_0 = T_cw_list[0]
    M_0 = T_cw_0[:3, :]  # Projection matrix for the first camera

    # Determine the camera pose with the largest baseline
    max_baseline = -np.inf
    T_cw_a = None
    pt_a = None
    for i in range(1, len(T_cw_list)):
        baseline = np.linalg.norm(T_cw_list[i][:3, 3] - T_cw_0[:3, 3])
        if baseline > max_baseline:
            max_baseline = baseline
            T_cw_a = T_cw_list[i]
            pt_a = pts_norm[i]
    M_1 = T_cw_a[:3, :]  # Projection matrix for the camera with the largest baseline

    # Triangulate the point using the Direct Linear Transform (DLT) method
    pt_w_h = cv2.triangulatePoints(M_0, M_1, pts_norm[0], pt_a)
    pt_w_h /= pt_w_h[3]  # Convert to Euclidean coordinates
    pt_w_est = pt_w_h.flatten()[:3]  # Estimated 3D point

    # Transformations for further refinement
    R_cw_a = T_cw_a[:3, :3]
    t_cw_a = T_cw_a[:3, 3]
    R_wc_a = R_cw_a.T
    t_wc_a = -R_wc_a @ t_cw_a
    pt_ca = R_cw_a @ pt_w_est + t_cw_a

    alpha, beta, rho = pt_ca[0] / pt_ca[2], pt_ca[1] / pt_ca[2], 1.0 / pt_ca[2]

    # Define cost function for optimization

    c = K_inv @ np.array([1., 1., .0])
    c_2 = c.T @ c

    def cost_function(params, *args):
        pts_norm, T_cw_list = args
        alpha, beta, rho = params

        n_measures = len(pts_norm)
        r = np.zeros(n_measures)

        for i, T_cw_i in enumerate(T_cw_list):
            z_meas = pts_norm[i]

            R_cw_i = T_cw_i[:3, :3]
            t_cw_i = T_cw_i[:3, 3]

            # compute model
            dR_ia = R_cw_i @ R_wc_a
            dt_ia = R_cw_i @ t_wc_a + t_cw_i
            p_ci = dR_ia @ np.array([alpha, beta, 1.]) + rho * dt_ia
            z_hat = p_ci[:2] / p_ci[2]

            # compute the error
            e = z_meas - z_hat

            # compute the norm for the Geman-McClure robust loss
            u = np.sqrt(e.T @ e)

            # Geman-McClure robust loss
            r[i] = 0.5 * c_2 * u ** 2 / (c_2 + u ** 2)

        return r

    initial_params = np.array([alpha, beta, rho])  # Initial guess of parameters
    args = (pts_norm, T_cw_list)  # Additional arguments needed for the cost function
    result = least_squares(cost_function, initial_params, args=args, method='lm')

    if not result.success:
        # print_warning("Optimization failed")
        return np.array([]), [], 0, 0, 1e6

    alpha, beta, rho = result.x

    u = np.linalg.norm(K_inv @ np.array([1., 1., .0]))
    c = 0.5 * c_2 * u ** 2 / (c_2 + u ** 2)
    # check if outlier
    if (result.fun == c).any() or rho <= 0.01:  # allow 1. pixel error
        return np.array([]), [], 0, 0, 1e6

    # Refine the 3D point estimate
    pt_w_est = np.linalg.inv(T_cw_a) @ np.array([alpha / rho, beta / rho, 1.0 / rho, 1.0])

    # Project points into all cameras
    refined_points_out = []
    sizes = []
    for T_cw_i in T_cw_list:
        pt_c = T_cw_i[:3, :] @ pt_w_est
        pt_c /= pt_c[2]
        sizes.append(np.linalg.norm(T_cw_i[:3, 3] - pt_w_est[:3]) * patch_size_half / K[0, 0])
        pt_proj = K @ pt_c
        refined_points_out.append(pt_proj[:2] / pt_proj[2])

    size = 2 * np.min(sizes)
    return pt_w_est[:3], refined_points_out, size, 0.0, 1e6


def adjust_pose(
        K: np.ndarray,
        points_3D: np.ndarray,
        points_2D: np.ndarray,
        reprojection_error: float,
        pose_ref: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adjusts the camera pose using 3D-2D correspondences and reprojection error.

    Args:
        K (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        points_3D (np.ndarray): Array of 3D points in world coordinates of shape (N, 3), where N is the number of points.
        points_2D (np.ndarray): Array of 2D points in image coordinates of shape (N, 2), corresponding to `points_3D`.
        reprojection_error (float): Maximum allowable reprojection error for pose estimation.
        pose_ref (bool, optional): Whether to refine the pose with additional iterations. Defaults to `False`.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - `T_star` (np.ndarray): Inverse transformation matrix of shape (4, 4), representing the camera-to-world pose.
            - `inliers` (np.ndarray): Boolean mask or array of inlier indices indicating the points used for the final pose.
    """
    # Ensure arrays are contiguous in memory for processing
    points_3D_ = np.ascontiguousarray(points_3D)
    points_2D_ = np.ascontiguousarray(points_2D)

    # Define the camera model parameters
    camera = {
        'model': 'PINHOLE',
        'width': int(K[0, 2] * 2),  # Image width inferred from intrinsic parameters
        'height': int(K[1, 2] * 2),  # Image height inferred from intrinsic parameters
        'params': [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]  # Focal lengths and principal point
    }

    # Estimate pose with or without refinement
    if pose_ref:
        pose, info = plib.estimate_absolute_pose(
            points_2D_, points_3D_, camera,
            {'max_reproj_error': reprojection_error, 'min_iterations': 100},
            {'max_iterations': 100}  # Allow refinement iterations
        )
    else:
        pose, info = plib.estimate_absolute_pose(
            points_2D_, points_3D_, camera,
            {'max_reproj_error': reprojection_error, 'min_iterations': 100},
            {'max_iterations': 0}  # No refinement iterations
        )

    # Compute the inverse transformation matrix
    T_star = np.linalg.inv(np.vstack((pose.Rt, [0., 0., 0., 1.])))

    return T_star, info['inliers']


def pose_error(T_est: np.ndarray, T_target: np.ndarray) -> Tuple[float, float]:
    """
    Computes the pose error between an estimated pose and the target pose.

    Args:
        T_est (np.ndarray): Estimated 4x4 transformation matrix.
        T_target (np.ndarray): Target 4x4 transformation matrix.

    Returns:
        tuple[float, float]:
            - `err_angle` (float): Angular error in degrees between the estimated and target rotation matrices.
            - `derr_dist` (float): Euclidean distance between the estimated and target translation vectors.
    """
    # Extract translation and rotation components from the estimated pose
    t_est = T_est[:3, 3]
    R_est = T_est[:3, :3]

    # Extract translation and rotation components from the target pose
    t_target = T_target[:3, 3]
    R_target = T_target[:3, :3]

    # Compute the Euclidean distance between translation vectors
    derr_dist = np.linalg.norm(t_est - t_target)

    # Compute the angular error between rotation matrices
    trace = np.trace(np.dot(R_est.T, R_target))
    angle_diff_radians = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    err_angle = np.abs(angle_diff_radians) * 180.0 / np.pi

    return err_angle, derr_dist


def rotation_error_with_sign(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Computes the signed rotation error in degrees between two rotation matrices.

    Args:
        R1 (np.ndarray): First rotation matrix (3x3).
        R2 (np.ndarray): Second rotation matrix (3x3).

    Returns:
        float: The signed angular error in degrees, ranging from -180° to 180°.
    """
    # Compute the relative rotation matrix
    R_error = np.dot(R1.T, R2)

    # Compute the rotation angle from the trace of the relative rotation matrix
    trace = np.trace(R_error)
    theta = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))  # Ensures stability of arccos

    # Check if the sine of the angle is close to 0 (e.g., 0 or π cases)
    if np.sin(theta) < 1e-6:
        # In such cases, the direction of rotation is ambiguous
        return theta * 180.0 / np.pi

    # Compute the axis of rotation from the skew-symmetric part of R_error
    axis = 1 / (2 * np.sin(theta)) * np.array([
        R_error[2, 1] - R_error[1, 2],
        R_error[0, 2] - R_error[2, 0],
        R_error[1, 0] - R_error[0, 1]
    ])

    # Normalize the axis and determine the sign of the angle
    if np.linalg.norm(axis) > 0:
        axis = axis / np.linalg.norm(axis)
        # If the first nonzero component of the axis is negative, flip the angle's sign
        if axis[0] < 0 or (axis[0] == 0 and (axis[1] < 0 or (axis[1] == 0 and axis[2] < 0))):
            theta = -theta

    # Convert the signed angle to degrees
    return theta * 180.0 / np.pi


def perturb_SE3(T: np.ndarray, sigma_translation: float = 0.1, sigma_rotation: float = np.pi / 18) -> np.ndarray:
    """
    Perturbs a 4x4 SE(3) transformation matrix by adding noise to the translation and rotation components.

    Args:
        T (np.ndarray): Input SE(3) transformation matrix (4x4) to be perturbed.
        sigma_translation (float): Standard deviation of the Gaussian noise applied to the translation vector.
        sigma_rotation (float): Standard deviation of the Gaussian noise (in radians) applied to the rotation vector.

    Returns:
        np.ndarray: Perturbed SE(3) transformation matrix (4x4).
    """
    # Extract translation and rotation from T
    translation = T[:3, 3]
    rotation = Rotation.from_matrix(T[:3, :3])

    # Perturb translation
    translation_perturbed = translation + np.random.normal(scale=sigma_translation, size=3)

    # Perturb rotation
    delta_rotation = Rotation.from_rotvec(np.random.normal(scale=sigma_rotation, size=3))
    rotation_perturbed = rotation * delta_rotation

    # Form the perturbed transformation matrix
    T_perturbed = np.eye(4)
    T_perturbed[:3, :3] = rotation_perturbed.as_matrix()
    T_perturbed[:3, 3] = translation_perturbed

    return T_perturbed


class IterativePnP:
    def __init__(self, model, K, reprojection_error, match_threshold, max_iter, tracker, visualization=False):
        self._initialize()
        self.model = model
        self.K = K
        self.reprojection_error = reprojection_error
        self.match_threshold = match_threshold
        self.max_iter = max_iter
        self.tracker = tracker
        self.visualization = visualization

    def _initialize(self):
        self.matched_landmarks = []
        self.target_kpts = []
        self.estimated_kpts = []
        self.scores = []
        self.camera_poses_list = []
        self.estimated_dist_errors = [[], [], []]
        self.estimated_angle_errors = [[], [], []]
        self.iterate = 0
        self.favor_estimates = [0, 0, 0]
        self.matches_per_iter = [0, 0, 0]

    def __call__(self, image, out_img, pose_gt, pose_prior):
        # Reset values
        # self._initialize()
        iterate = 0

        cam_poses_list = []
        cam_poses_list.append(copy.deepcopy(pose_gt))
        cam_poses_list.append(copy.deepcopy(pose_prior))

        matched_landmarks_out = []

        while iterate < self.max_iter:
            iterate += 1
            # Match points
            matched_rendered_kpts, matched_target_kpts, matched_landmarks, rendered_pts, rendered_landmasks, scores = matcher_fast(
                self.model, image, self.tracker, self.K, pose_prior, self.match_threshold)

            # Early stop if no matches found
            if len(matched_landmarks) == 0:
                print_warning(f"Early stop. No matches found.")
                for _ in range(self.max_iter - iterate):
                    self.estimated_dist_errors[iterate - 1].append(np.inf)
                    self.estimated_angle_errors[iterate - 1].append(np.inf)
                    self.iterate += 1
                break

            # Pose adjustment using matched points
            T_star, inliers = adjust_pose(self.K, points_3D=matched_landmarks, points_2D=matched_target_kpts,
                                          reprojection_error=self.reprojection_error, pose_ref=iterate == self.max_iter)

            matched_target_kpts = matched_target_kpts[inliers]
            matched_rendered_kpts = matched_rendered_kpts[inliers]
            matched_landmarks = matched_landmarks[inliers]

            # Visualization of matches
            if self.visualization:
                out_img = visualize_matches(image, out_img, matched_rendered_kpts, matched_target_kpts,
                                            rendered_landmasks,
                                            pose_prior, self.K)
            inliers = np.sum(np.array(inliers))
            self.matches_per_iter[iterate - 1] += inliers
            if inliers <= 3:
                print_warning(f"Early stop. Pose not found.")
                for _ in range(self.max_iter - iterate):
                    self.estimated_dist_errors[iterate - 1].append(np.inf)
                    self.estimated_angle_errors[iterate - 1].append(np.inf)
                    self.iterate += 1
                break

            # Update results
            matched_landmarks_out.append(matched_landmarks)
            self.target_kpts.append(matched_target_kpts)
            self.estimated_kpts.append(matched_rendered_kpts)
            self.scores.append(scores)

            # Update pose and error calculations
            err_angle, err_dist = pose_error(T_star, pose_gt)
            self.estimated_dist_errors[iterate - 1].append(err_dist)
            self.estimated_angle_errors[iterate - 1].append(err_angle)

            cam_poses_list.append(copy.deepcopy(T_star))
            self.favor_estimates[iterate - 1] += 1

        self.matched_landmarks.append(matched_landmarks_out)
        self.camera_poses_list.append(cam_poses_list)
