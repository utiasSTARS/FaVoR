#  Copyright (c) 2023 Vincenzo Polizzi <vincenzo dot polizzi at mail dot utoronto dot ca>
#  (Space and Terrestrial Autonomous Robotic Systems Laboratory, University of Toronto,
#  Institute for Aerospace Studies).
#  This file is subject to the terms and conditions defined in the file
#  'LICENSE', which is part of this source code package.

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import lib.cuda
from lib.utils_favor.transform_utils import CV2B

from lib.models.voxel_model import VoxelModel


class FaVoRmodel(nn.Module):
    def __init__(self, voxels_args, **kwargs):
        """
        Initialize the FaVoRmodel class.

        Args:
            voxels_args (list): List of arguments for initializing voxel models.
            kwargs: Additional arguments passed to the VoxelModel.

        Attributes:
            voxels (nn.ModuleList): A list of VoxelModel instances.
            pts (numpy.ndarray): Array of point weights in homogeneous coordinates.
            channels (int): Number of color channels from the first voxel.
            time_eval (bool): Toggle for evaluation mode.
            prev_descs (NoneType): Placeholder for previously computed descriptors.
        """
        super(FaVoRmodel, self).__init__()
        self.voxels = nn.ModuleList([])
        self.pts = []

        # Initialize voxel models and their corresponding point weights
        for v in voxels_args:
            # Append point weights in homogeneous coordinates (add 1.0 for w)
            self.pts.append(np.concatenate([v['point_w'], [1.0]]))
            # Create VoxelModel instances and add to voxels list
            self.voxels.append(VoxelModel(**v, **kwargs))

        # Stack all point weights into a single numpy array
        self.pts = np.stack(self.pts)
        self.channels = self.voxels[0].channels  # Assuming all voxels share the same number of channels

        self.time_eval = False
        self.prev_descs = None

    def top_n(self, N: int, psnr=False):
        """
        Retain the top-N voxel models based on PSNR or the number of views.

        Args:
            N (int): Number of voxel models to keep.
            psnr (bool): If True, sort by PSNR; otherwise, sort by the number of views.
        """
        if psnr:
            # Sort by PSNR in descending order and keep the top N trained voxels
            indices = np.argsort([v.psnr for v in self.voxels if v.trained])[::-1][:N]
        else:
            # Sort by number of views in descending order and keep the top N trained voxels
            indices = np.argsort([v.n_of_views for v in self.voxels if v.trained])[::-1][:N]

        # Retain the top-N voxels and their corresponding points
        self.voxels = nn.ModuleList([self.voxels[i] for i in indices])
        self.pts = self.pts[indices]

    def get_n_voxels(self) -> int:
        """
        Get the number of voxel models.

        Returns:
            int: Number of voxel models.
        """
        return len(self.voxels)

    def change_time_eval(self):
        """
        Toggle the evaluation mode for time-based processing.
        """
        self.time_eval = not self.time_eval

    def get_kwargs(self):
        """
        Retrieve the arguments required to reconstruct the model.

        Returns:
            dict: Contains the arguments for each voxel model.
        """
        return {
            'voxels_args': [v.get_kwargs() for v in self.voxels],
        }

    def get_bboxes(self):
        """
        Get the bounding boxes of trained voxel models.

        Returns:
            list: List of dictionaries containing the min and max coordinates of voxels.
        """
        return [{'xyz_min': v.xyz_min, 'xyz_max': v.xyz_max} for v in self.voxels if v.trained]

    def point2B(self, p):
        """
        Convert a point to the transformation matrix's translation vector.

        Args:
            p (numpy.ndarray): Point in 3D space.

        Returns:
            numpy.ndarray: Translation vector in 3D space.
        """
        rot = np.diag([1, 1, 1])  # Identity rotation matrix
        T_pts_w = np.column_stack((rot, p))  # Add translation column
        T_pts_w = np.row_stack((T_pts_w, [0, 0, 0, 1]))  # Homogeneous coordinates
        T_pts_w = CV2B(T_pts_w)  # Convert to specific format
        return T_pts_w[:3, 3]

    def project_points(self, K, c2w, w, h):
        """
        Project 3D points into 2D image space.

        Args:
            K (numpy.ndarray): Intrinsic camera matrix.
            c2w (numpy.ndarray): Camera-to-world transformation matrix.
            w (int): Image width.
            h (int): Image height.

        Returns:
            tuple: Projected 2D keypoints and visibility mask.
        """
        mask = np.ones(len(self.pts), dtype=bool)  # Initialize visibility mask
        kpts_cam = np.linalg.inv(c2w)[:3, :] @ self.pts.T  # Transform points to camera space
        mask &= kpts_cam[2, :] > 0  # Keep points in front of the camera
        kps = K @ kpts_cam  # Project points using camera intrinsics
        kps = kps[:2] / kps[2]  # Normalize by depth
        kps = kps.transpose()
        # Ensure points are within image bounds with a 5-pixel margin
        mask &= (kps[:, 0] > 5) & (kps[:, 1] > 5) & (kps[:, 0] < h - 5) & (kps[:, 1] < w - 5)
        return kps, mask

    def forward(self, K, p_wc, h, w, device='cuda'):
        """
        Perform forward pass for rendering.

        Args:
            K (numpy.ndarray): Camera intrinsics.
            p_wc (numpy.ndarray): World-to-camera transformation matrix.
            h (int): Image height.
            w (int): Image width.
            device (str): Device for computation.

        Returns:
            tuple: Descriptors, rendered points, and landmarks.
        """
        return self.point_render(K, p_wc, h, w, device)

    @torch.no_grad()
    def render(self, K, c2w, near, stepsize, half_patch_size, h, w, normalize=True):
        """
        Render the scene.

        Args:
            K, c2w, near, stepsize, half_patch_size, h, w, normalize: Rendering parameters.

        Returns:
            Rendering results.
        """
        return self(K, c2w, near, stepsize, half_patch_size, h, w, normalize=normalize)

    @torch.no_grad()
    def point_render(self, K, p_wc, h, w, device='cuda'):
        """
        Render points using ray tracing.

        Args:
            K (numpy.ndarray): Camera intrinsics.
            p_wc (numpy.ndarray): World-to-camera transformation matrix.
            h (int): Image height.
            w (int): Image width.
            device (str): Device for computation.

        Returns:
            tuple: Normalized descriptors, rendered 2D points, and landmarks.
        """
        visible_pts, mask = self.project_points(K, p_wc, h, w)

        # Extract visible 3D landmarks and compute ray directions
        visible_landmarks = self.pts[mask][:, :3]
        rays_d = visible_landmarks - p_wc[:3, 3]  # Direction from camera to point
        rays_d /= np.linalg.norm(rays_d, axis=1)[:, None]  # Normalize rays

        rays_o = np.broadcast_to(p_wc[:3, 3], rays_d.shape)  # Camera origin

        # Convert rays to tensors for computation
        rays_d = torch.tensor(rays_d, device=device, dtype=torch.float32)
        rays_o = torch.tensor(rays_o, device=device, dtype=torch.float32)

        # Prepare storage for descriptors and landmarks
        descs = torch.zeros(len(mask), self.channels, device=device)
        rendered_pts = np.zeros((len(mask), 2))
        landmasks = np.zeros((len(mask), 3))

        v_ids = np.arange(len(mask))[mask]
        for i, idx in enumerate(v_ids):
            pt = visible_pts[idx]
            land = visible_landmarks[i]
            rendered_pts[i] = pt
            landmasks[i] = land
            # Render descriptors using voxel models
            render_result = self.voxels[idx](rays_o[i].view(-1, 3), rays_d[i].view(-1, 3))
            descs[i] = render_result['desc'].flatten()

        # Normalize descriptors
        descs = F.normalize(descs, p=2, dim=1).detach().cpu().numpy()

        if len(descs) == 0:
            return np.array([]), np.array([]), np.array([])

        return descs, rendered_pts, landmasks
