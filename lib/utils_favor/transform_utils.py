#  Copyright (c) 2023 Vincenzo Polizzi <vincenzo dot polizzi at mail dot utoronto dot ca>
#  (Space and Terrestrial Autonomous Robotic Systems Laboratory, University of Toronto,
#  Institute for Aerospace Studies).
#  This file is subject to the terms and conditions defined in the file
#  'LICENSE', which is part of this source code package.

import numpy as np
import copy


# Some similar to https://github.com/NVlabs/instant-ngp/blob/master/scripts/colmap2nerf.py#L350

def B2CV(M0: np.ndarray) -> np.ndarray:
    """
    Converts a transformation matrix from Blender coordinates to OpenCV coordinates.

    Args:
        M0 (np.ndarray): The input transformation matrix in Blender coordinates (4x4).

    Returns:
        np.ndarray: The equivalent transformation matrix in OpenCV coordinates (4x4).
    """
    T = copy.deepcopy(M0)
    T[0:3, 2] *= -1  # Flip the y and z axes
    T[0:3, 1] *= -1
    T = T[[1, 0, 2, 3], :]  # Swap y and z axes
    T[2, :] *= -1  # Flip the world upside down
    return T


def CV2B(M0: np.ndarray) -> np.ndarray:
    """
    Converts a transformation matrix from OpenCV coordinates to Blender coordinates.

    Args:
        M0 (np.ndarray): The input transformation matrix in OpenCV coordinates (4x4).

    Returns:
        np.ndarray: The equivalent transformation matrix in Blender coordinates (4x4).
    """
    T = copy.deepcopy(M0)
    T[2, :] *= -1  # Reverse flipping the world upside down
    T = T[[1, 0, 2, 3], :]  # Reverse swapping y and z axes
    T[0:3, 1] *= -1  # Reverse flipping the y axis
    T[0:3, 2] *= -1  # Reverse flipping the z axis
    return T


def O3D2B(pose_: np.ndarray) -> np.ndarray:
    """
    Converts a transformation matrix from Open3D coordinates to Blender coordinates.

    Args:
        pose_ (np.ndarray): The input transformation matrix in Open3D coordinates (4x4).

    Returns:
        np.ndarray: The equivalent transformation matrix in Blender coordinates (4x4).
    """
    pose = copy.deepcopy(pose_)
    R_y = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])

    R_z = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])  # Reflection matrix

    pose[:3, :3] = R_y @ R_z @ pose[:3, :3]
    return pose


def B2O3D(pose_: np.ndarray) -> np.ndarray:
    """
    Converts a transformation matrix from Blender coordinates to Open3D coordinates.

    Args:
        pose_ (np.ndarray): The input transformation matrix in Blender coordinates (4x4).

    Returns:
        np.ndarray: The equivalent transformation matrix in Open3D coordinates (4x4).
    """
    pose = copy.deepcopy(pose_)
    R_y = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])

    R_z = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])  # Reflection matrix

    pose[:3, :3] = R_z @ R_y @ pose[:3, :3]
    return pose


def CV2O3D(pose_: np.ndarray) -> np.ndarray:
    """
    Converts a transformation matrix from OpenCV coordinates to Open3D coordinates via Blender.

    Args:
        pose_ (np.ndarray): The input transformation matrix in OpenCV coordinates (4x4).

    Returns:
        np.ndarray: The equivalent transformation matrix in Open3D coordinates (4x4).
    """
    return B2O3D(CV2B(pose_))
