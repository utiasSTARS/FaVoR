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
from lib.models.grid import DenseGrid, MaskGrid


class VoxelModel(nn.Module):
    """
    A PyTorch-based neural network model for representing and rendering a voxel grid,
    typically used in volumetric rendering or 3D scene reconstruction tasks.
    """

    def __init__(self,
                 xyz_min,  # Minimum coordinates for the voxel grid
                 xyz_max,  # Maximum coordinates for the voxel grid
                 num_voxels=0,  # Total number of voxels in the grid
                 alpha_init=None,  # Initial alpha value for the grid
                 mask_cache_path=None,  # Path to cache for pre-computed mask grid
                 mask_cache_thres=1e-3,  # Threshold for mask cache filtering
                 mask_cache_world_size=None,  # Size of the world grid for mask cache
                 fast_color_thres=0,  # Fast color threshold for skipping low-opacity regions
                 vox_id=-1,  # Voxel ID (used for identification)
                 channels=3,  # Number of channels (typically for color representation)
                 point_w=None,  # Point weight (if applicable)
                 density_config={},  # Configuration dictionary for density grid
                 k0_config={},  # Configuration dictionary for color grid (k0)
                 trained=False,  # Boolean flag indicating if the model has been trained
                 psnr=0,  # Peak Signal-to-Noise Ratio (used for evaluation)
                 images_seen=[],  # List of images seen by the model
                 n_of_views=0,  # Number of views in the dataset
                 **kwargs):
        super(VoxelModel, self).__init__()

        # Initialize model parameters
        self.n_of_views = n_of_views
        self.images_seen = images_seen
        self.psnr = psnr
        self.trained = trained
        self.point_w = point_w
        self.channels = channels
        self.vox_id = vox_id
        self.fast_color_thres = fast_color_thres

        # Register voxel grid boundaries as buffers
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))

        # Initialize density and color grid
        self.alpha_init = alpha_init
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1 / (1 - alpha_init) - 1)]))
        self._set_grid_resolution(num_voxels)

        # Density Grid (for voxel space)
        self.density_config = density_config
        self.density = DenseGrid(channels=1, world_size=self.world_size,
                                 xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                                 config=self.density_config)

        # Color Grid (for voxel space)
        self.k0_config = k0_config
        self.k0 = DenseGrid(channels=self.channels, world_size=self.world_size,
                            xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                            config=self.k0_config)

        # Mask Cache for Free Space
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        self._initialize_mask_cache(mask_cache_world_size)

    def _initialize_mask_cache(self, mask_cache_world_size):
        """
        Initializes the mask cache grid to store precomputed masks for free space.
        """
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size

        if self.mask_cache_path:
            # Load mask cache if path is provided
            mask_cache = MaskGrid(vox_id=self.vox_id, path=self.mask_cache_path,
                                  mask_cache_thres=self.mask_cache_thres).to(self.xyz_min.device)
            mask = mask_cache(self._generate_grid_xyz(mask_cache_world_size))
        else:
            # Otherwise, use a default mask grid (all space is considered free)
            mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)

        self.mask_cache = MaskGrid(vox_id=self.vox_id, path=None, mask=mask,
                                   xyz_min=self.xyz_min, xyz_max=self.xyz_max)

    def _generate_grid_xyz(self, mask_cache_world_size):
        """
        Generates a 3D grid of XYZ coordinates within the bounds of the voxel grid.
        """
        return torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], mask_cache_world_size[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], mask_cache_world_size[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], mask_cache_world_size[2]),
        ), -1)

    def _set_grid_resolution(self, num_voxels):
        """
        Sets the resolution of the voxel grid based on the number of voxels.
        """
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels).pow(1 / 3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = 1.

    def get_kwargs(self):
        """
        Returns model parameters as a dictionary.
        """
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
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

    def is_training(self):
        """
        Sets the model's training status to True.
        """
        self.trained = True

    def voxel_count_views(self, rays_o_tr, rays_d_tr, imsz, near, stepsize, downrate=1, irregular_shape=False):
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
        return count

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        """
        Scales the voxel grid resolution and updates the density and color grids.
        """
        self._set_grid_resolution(num_voxels)
        self.density.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)

        if np.prod(self.world_size.tolist()) <= 256 ** 3:
            self._update_mask_cache()

    @torch.no_grad()
    def _update_mask_cache(self):
        """
        Updates the mask cache based on new grid resolution.
        """
        self_grid_xyz = self._generate_grid_xyz(self.world_size)
        self_alpha = \
            F.max_pool3d(self.activate_density(self.density.get_dense_grid()), kernel_size=3, padding=1, stride=1)[0, 0]
        self.mask_cache = MaskGrid(path=None,
                                   mask=self.mask_cache(self_grid_xyz) & (self_alpha > self.fast_color_thres),
                                   xyz_min=self.xyz_min, xyz_max=self.xyz_max)

    @torch.no_grad()
    def update_occupancy_cache(self):
        """
        Updates the occupancy mask based on the latest voxel grid data.
        """
        cache_grid_xyz = self._generate_grid_xyz(self.mask_cache.mask.shape)
        cache_grid_density = self.density(cache_grid_xyz)[None, None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0, 0]
        self.mask_cache.mask &= (cache_grid_alpha > self.fast_color_thres)

    def density_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / self.channels
        self.density.total_variation_add_grad(w, w, w, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / self.channels
        self.k0.total_variation_add_grad(w, w, w, dense_mode)
        
    def activate_density(self, density, interval=None):
        """
        Converts raw density values to alpha values using a custom activation function.
        """
        return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(density.shape)

    def sample_ray(self, rays_o, rays_d):
        """
        Samples points along a ray within the voxel grid, using the ray's origin and direction.
        """
        stepsize = 0.1
        near = 0.2
        far = 1e9
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.voxel_size

        # Call external function to sample points along rays
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = torch.ops.render_utils.sample_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)

        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        interpx = (rays_o[ray_id] - ray_pts).norm(p=2, dim=-1, keepdim=False)

        return ray_pts, ray_id, step_id, interpx

    def forward(self, rays_o, rays_d):
        """
        Main function for forward pass in volume rendering. This calculates the ray intersections
        with the voxel grid, applying alpha and color values based on the density grid.

        Args:
            rays_o (torch.Tensor): Ray origins in world coordinates (size: [N, 3])
            rays_d (torch.Tensor): Ray directions in world coordinates (size: [N, 3])

        Returns:
            dict: Dictionary containing the following output data:
                - 'alphainv_last' (torch.Tensor): Inverse alpha value for the last sampled point
                - 'weights' (torch.Tensor): Accumulated transmittance weights for volume rendering
                - 'desc' (torch.Tensor): Accumulated color values for volume rendering
                - 'depth' (torch.Tensor): Accumulated depth values for volume rendering
                - 'ray_id' (torch.Tensor): Ray ID for each sampled point
                - 'N' (int): Number of rays
                - 'n_max' (int): Maximum number of sampled points
                - 't' (torch.Tensor): Interpolation values for each sampled point
                - 's' (torch.Tensor): Depth values for each sampled point
        """
        assert len(rays_o.shape) == 2 and rays_o.shape[-1] == 3, 'Only support point queries in [N, 3] format'
        N = len(rays_o)

        # Sample points along rays
        ray_pts, ray_id, step_id, t = self.sample_ray(rays_o=rays_o, rays_d=rays_d)
        interval = 0.1 * 1.0  # self.voxel_size_ratio

        # Apply mask cache to skip known free space
        if self.mask_cache is not None:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            t = t[mask]

        # Query alpha (density) and color (k0) for sampled points
        density = self.density(ray_pts)
        alpha = self.activate_density(density, interval)

        # Apply fast color threshold to filter points
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
            t = t[mask]

        # Compute accumulated transmittance and color for volume rendering
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)

        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            t = t[mask]

        # Compute depth and color
        s = 1 - 1 / (1 + t)  # [0, inf] => [0, 1]
        desc = self.k0(ray_pts)

        # Segment the results to obtain final rendered output
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
            'n_max': len(t),
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
