#  Copyright (c) 2023 Vincenzo Polizzi <vincenzo dot polizzi at mail dot utoronto dot ca>
#  (Space and Terrestrial Autonomous Robotic Systems Laboratory, University of Toronto,
#  Institute for Aerospace Studies).
#  This file is subject to the terms and conditions defined in the file
#  'LICENSE', which is part of this source code package.

import numpy as np
import cv2
from typing import Tuple, List

from scipy.optimize import least_squares


def mnn_matcher(desc1: np.ndarray, desc2: np.ndarray, match_thr: float = 0.9) -> np.ndarray:
    """
    Finds matches between two descriptor lists (desc1 and desc2) using mutual nearest neighbor (MNN) matching.

    Args:
        desc1 (np.ndarray): Descriptor array from the first set of keypoints.
        desc2 (np.ndarray): Descriptor array from the second set of keypoints.
        match_thr (float): Threshold for similarity score (default: 0.9).

    Returns:
        np.ndarray: Array of matches with shape (N, 2), where each row is a pair of matched indices.
    """
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return np.zeros((0, 2), dtype=int)

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

    return matches


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
