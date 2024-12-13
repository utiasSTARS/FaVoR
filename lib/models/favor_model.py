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
from .utils_favor.transform_utils import CV2B

from voxel_model import VoxelModel


class FaVoRmodel(nn.Module):
    def __init__(self, voxels_args, **kwargs):
        super(FaVoRmodel, self).__init__()
        self.voxels = nn.ModuleList([])
        self.pts = []
        for v in voxels_args:
            # self.pts.append(v['point_w'])
            self.pts.append(np.concatenate([v['point_w'], [1.0]]))
            self.voxels.append(
                VoxelModel(**v,
                           **kwargs))

        self.pts = np.stack(self.pts)  # .transpose()
        self.channels = self.voxels[0].channels

        self.time_eval = False
        self.prev_descs = None

    def top_n(self, N: int, psnr=False):
        # limit number of voxels and points to N
        # get the indices of the first N voxels with high PSNR or number of views
        if psnr:
            indices = np.argsort([v.psnr for v in self.voxels if v.trained])[::-1][:N]
            self.voxels = nn.ModuleList([self.voxels[i] for i in indices])
            self.pts = self.pts[indices]
        else:
            indices = np.argsort([v.n_of_views for v in self.voxels if v.trained])[::-1][:N]
            self.voxels = nn.ModuleList([self.voxels[i] for i in indices])
            self.pts = self.pts[indices]

    def change_time_eval(self):
        self.time_eval = not self.time_eval

    def get_kwargs(self):
        return {
            'voxels_args': [v.get_kwargs() for v in self.voxels],
        }

    def get_bboxes(self):
        return [{'xyz_min': v.xyz_min, 'xyz_max': v.xyz_max} for v in self.voxels if v.trained]

    def point2B(self, p):
        rot = np.diag([1, 1, 1])
        T_pts_w = np.column_stack((rot, p))
        T_pts_w = np.row_stack((T_pts_w, [0, 0, 0, 1]))
        T_pts_w = CV2B(T_pts_w)
        return T_pts_w[:3, 3]

    def project_points(self, K, c2w, w, h):
        mask = np.ones(len(self.pts), dtype=bool)
        kpts_cam = np.linalg.inv(c2w)[:3, :] @ self.pts.T
        mask &= kpts_cam[2, :] > 0
        kps = K @ kpts_cam
        kps = kps[:2] / kps[2]
        kps = kps.transpose()
        mask &= (kps[:, 0] > 5) & (kps[:, 1] > 5) & (kps[:, 0] < h - 5) & (kps[:, 1] < w - 5)
        return kps, mask

    def forward(self, K, p_wc, h, w, device='cuda'):
        return self.point_render(K, p_wc, h, w, device)

    @torch.no_grad()
    def render(self, K, c2w, near, stepsize, half_patch_size, h, w, normalize=True):
        return self(K, c2w, near, stepsize, half_patch_size, h, w, normalize=normalize)

    @torch.no_grad()
    def point_render(self, K, p_wc, h, w, device='cuda'):

        # this part of the points projection is used for LightGlue (only)
        visible_pts, mask = self.project_points(K, p_wc, h, w)

        # dirs from poses and points directly
        # len_tot = 100
        visible_landmarks = self.pts[mask][:, :3]
        # visible_landmarks = visible_landmarks[:len_tot]

        rays_d = visible_landmarks - p_wc[:3, 3]
        # normalize in numpy
        rays_d = rays_d / np.linalg.norm(rays_d, axis=1)[:, None]

        rays_o = np.broadcast_to(p_wc[:3, 3], rays_d.shape)

        rays_d = torch.tensor(rays_d, device=device, dtype=torch.float32)
        rays_o = torch.tensor(rays_o, device=device, dtype=torch.float32)

        descs = torch.zeros(len(mask), self.channels, device=device)  # len_tot
        rendered_pts = np.zeros((len(mask), 2))  # len_tot
        landmasks = np.zeros((len(mask), 3))  # len_tot
        v_ids = np.arange(len(mask))[mask]
        # v_ids = v_ids[:len_tot]
        for i, idx in enumerate(v_ids):
            pt = visible_pts[idx]
            land = visible_landmarks[i]
            rendered_pts[i] = pt
            landmasks[i] = land
            render_result = self.voxels[idx](rays_o[i].view(-1, 3), rays_d[i].view(-1, 3))

            descs[i] = render_result['desc'].flatten()
        del rays_o, rays_d

        # normalize all descs p=2
        descs = F.normalize(descs, p=2, dim=1).detach().cpu().numpy()

        if len(descs) == 0:
            return np.array([]), np.array([]), np.array([])

        return descs, rendered_pts, landmasks
