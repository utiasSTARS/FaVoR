#  Copyright (c) 2023 Vincenzo Polizzi <vincenzo dot polizzi at mail dot utoronto dot ca>
#  (Space and Terrestrial Autonomous Robotic Systems Laboratory, University of Toronto,
#  Institute for Aerospace Studies).
#  This file is subject to the terms and conditions defined in the file
#  'LICENSE', which is part of this source code package.

import numpy as np
import cv2
from typing import Tuple, List


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
        target_map: np.ndarray, pt: Tuple[int, int], patch: int, hwc: bool = False) -> np.ndarray:
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

    # Extract patch based on format
    if hwc:
        return target_map[r_x_0:r_x_1, r_y_0:r_y_1, ...]

    return target_map[..., r_x_0:r_x_1, r_y_0:r_y_1]
