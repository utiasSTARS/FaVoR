#  Copyright (c) 2023 Vincenzo Polizzi <vincenzo dot polizzi at mail dot utoronto dot ca>
#  (Space and Terrestrial Autonomous Robotic Systems Laboratory, University of Toronto,
#  Institute for Aerospace Studies).
#  This file is subject to the terms and conditions defined in the file
#  'LICENSE', which is part of this source code package.

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import segment_coo
from lib.grid import DenseGrid, MaskGrid

import lib.cuda
from .utils_favor.transform_utils import CV2B


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


class VoxelModel(nn.Module):

    def __init__(self,
                 xyz_min,
                 xyz_max,
                 num_voxels=0,
                 num_voxels_base=0,
                 alpha_init=None,
                 mask_cache_path=None,
                 mask_cache_thres=1e-3,
                 mask_cache_world_size=None,
                 fast_color_thres=0,
                 vox_id=-1,
                 channels=3,
                 point_w=None,
                 density_config={},
                 k0_config={},
                 trained=False,
                 psnr=0,
                 images_seen=[],
                 n_of_views=0,
                 **kwargs):
        super(VoxelModel, self).__init__()
        self.n_of_views = n_of_views
        self.images_seen = images_seen
        self.psnr = psnr
        self.trained = trained
        self.point_w = point_w
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))

        self.fast_color_thres = fast_color_thres
        self.vox_id = vox_id
        self.channels = channels

        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1 / 3)

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1 / (1 - alpha_init) - 1)]))
        # print('dvgo: set density bias shift to', self.act_shift)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)

        # init density voxel grid
        self.density_config = density_config
        self.density = DenseGrid(channels=1, world_size=self.world_size,
                                 xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                                 config=self.density_config)

        # init color representation
        self.k0_config = k0_config
        self.k0 = DenseGrid(channels=self.channels, world_size=self.world_size,
                            xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                            config=self.k0_config)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid (2021/1/31)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size

        if mask_cache_path is not None and mask_cache_path:
            mask_cache = MaskGrid(
                vox_id=vox_id,
                path=mask_cache_path,
                mask_cache_thres=mask_cache_thres).to(self.xyz_min.device)
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], mask_cache_world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], mask_cache_world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], mask_cache_world_size[2]),
            ), -1)
            mask = mask_cache(self_grid_xyz)
        else:
            mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)

        self.mask_cache = MaskGrid(
            vox_id=vox_id,
            path=None, mask=mask,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max)

    def is_training(self):
        self.trained = True

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1 / 3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'voxel_size_ratio': self.voxel_size_ratio,
            'vox_id': self.vox_id,
            'point_w': self.point_w,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            'mask_cache_world_size': list(self.mask_cache.mask.shape),
            'fast_color_thres': self.fast_color_thres,
            'density_config': self.density_config,
            'k0_config': self.k0_config,
            'channels': self.channels,
            'trained': self.trained,
            'psnr': self.psnr,
            'images_seen': self.images_seen
        }

    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near_clip):
        # maskout grid points that between cameras and their near planes
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
        ), -1)
        nearest_dist = torch.stack([
            (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
            for co in cam_o.split(100)  # for memory saving
        ]).amin(0)
        self.density.grid[nearest_dist[None, None] <= near_clip] = -100

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        self._set_grid_resolution(num_voxels)

        self.density.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)

        if np.prod(self.world_size.tolist()) <= 256 ** 3:
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
            ), -1)
            self_alpha = \
                F.max_pool3d(self.activate_density(self.density.get_dense_grid()), kernel_size=3, padding=1, stride=1)[
                    0, 0]
            self.mask_cache = MaskGrid(
                path=None, mask=self.mask_cache(self_grid_xyz) & (self_alpha > self.fast_color_thres),
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)


    @torch.no_grad()
    def update_occupancy_cache(self):
        cache_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2]),
        ), -1)
        cache_grid_density = self.density(cache_grid_xyz)[None, None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0, 0]
        self.mask_cache.mask &= (cache_grid_alpha > self.fast_color_thres)

    def voxel_count_views(self, rays_o_tr, rays_d_tr, imsz, near, stepsize, downrate=1, irregular_shape=False):
        # print('dvgo: voxel_count_views start')
        # eps_time = time.time()
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        N_samples = int(np.linalg.norm(np.array(self.world_size.cpu()) + 1) / stepsize) + 1
        rng = torch.arange(N_samples)[None].float()
        count = torch.zeros_like(self.density.get_dense_grid())
        device = rng.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            if irregular_shape:
                rays_o_ = rays_o_.split(10000)
                rays_d_ = rays_d_.split(10000)
            else:
                rays_o_ = rays_o_[::downrate, ::downrate].to(device).flatten(0, -2).split(10000)
                rays_d_ = rays_d_[::downrate, ::downrate].to(device).flatten(0, -2).split(10000)

            for rays_o, rays_d in zip(rays_o_, rays_d_):
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.xyz_max - rays_o) / vec
                rate_b = (self.xyz_min - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
                # t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
                step = stepsize * self.voxel_size * rng
                interpx = (t_min[..., None] + step / rays_d.norm(dim=-1, keepdim=True))
                rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
                ones(rays_pts).sum().backward()
            with torch.no_grad():
                count += (ones.grid.grad > 1)
        # eps_time = time.time() - eps_time
        # print('dvgo: voxel_count_views finish (eps time:', eps_time, 'sec)')
        return count

    def density_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / self.channels
        self.density.total_variation_add_grad(w, w, w, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / self.channels
        self.k0.total_variation_add_grad(w, w, w, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)

    def sample_ray(self, rays_o, rays_d):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        stepsize = 0.1
        near = 0.2
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = torch.ops.render_utils.sample_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        interpx = (rays_o[ray_id] - ray_pts).norm(p=2, dim=-1, keepdim=False)

        return ray_pts, ray_id, step_id, interpx

    def forward(self, rays_o, rays_d):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        '''
        assert len(rays_o.shape) == 2 and rays_o.shape[-1] == 3, 'Only support point queries in [N, 3] format'

        N = len(rays_o)
        stepsize = 0.1

        # sample points on rays
        ray_pts, ray_id, step_id, t = self.sample_ray(
            rays_o=rays_o, rays_d=rays_d)
        interval = stepsize * self.voxel_size_ratio

        n_max = len(t)
        # skip known free space
        if self.mask_cache is not None:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            t = t[mask]

        # query for alpha w/ post-activation
        density = self.density(
            ray_pts)  # rays are nomralized into density, same for k0, better to have them already normalized
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            # density = density[mask]
            alpha = alpha[mask]
            t = t[mask]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            # alpha = alpha[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            t = t[mask]

        s = 1 - 1 / (1 + t)  # [0, inf] => [0, 1]

        # desc_raw = self.k0(ray_pts)
        # desc = torch.tanh(desc_raw)
        # desc = torch.relu(desc_raw + 1.) - 1
        desc = self.k0(ray_pts)

        desc_marched = (segment_coo(
            src=(weights.unsqueeze(-1) * desc),
            index=ray_id,
            out=torch.zeros([N, self.channels]),
            reduce='sum'
        ) + alphainv_last.unsqueeze(-1)).clamp(min=-1, max=1)

        depth = segment_coo(
            src=(weights * step_id),
            index=ray_id,
            out=torch.zeros([N]),
            reduce='sum')

        return {
            'alphainv_last': alphainv_last,
            'weights': weights,
            'desc': desc_marched,
            'depth': depth,
            'ray_id': ray_id,
            'N': N,
            'n_max': n_max,
            't': t,
            's': s,
        }


class Raw2Alpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        '''
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        '''
        exp, alpha = torch.ops.render_utils.raw2alpha(density, shift, interval)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        '''
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)'
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        '''
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return torch.ops.render_utils.raw2alpha_backward(exp, grad_back.contiguous(), interval), None, None


class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = torch.ops.render_utils.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = torch.ops.render_utils.alpha2weight_backward(
            alpha, weights, T, alphainv_last,
            i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None
