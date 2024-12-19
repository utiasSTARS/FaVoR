#  Copyright (c) 2023 Vincenzo Polizzi <vincenzo dot polizzi at mail dot utoronto dot ca>
#  (Space and Terrestrial Autonomous Robotic Systems Laboratory, University of Toronto,
#  Institute for Aerospace Studies).
#  This file is subject to the terms and conditions defined in the file
#  'LICENSE', which is part of this source code package.

import argparse
import os
import random
import sys
import time
import mmengine
import numpy as np
import glob
import torch
from typing import Tuple, List

from lib.data_loaders.CambridgeDataloader import CambridgeDataloader
from lib.data_loaders.SevScenesDataloader import SevScenesDataloader
from lib.data_loaders.Dataloader import Dataloader
from lib.optimizer.masked_adam import MaskedAdam
from lib.models.favor_model import FaVoRmodel
from lib.trackers.ALIKE_Tracker import AlikeTracker
from lib.trackers.SuperPoint_Tracker import SuperPointTracker
from lib.trackers.base_tracker import BaseTracker
from lib.utils_favor.log_utils import print_info, print_warning, print_success, print_error


def model2channels(net_model: str) -> int:
    """Get the number of output channels for a given model.

    Args:
        net_model (str): Name of the model.

    Returns:
        int: Number of output channels.

    Raises:
        ValueError: If the model name is not supported.
    """
    model_channels = {
        'alike-n': 128,
        'alike-l': 128,
        'alike-t': 64,
        'alike-s': 94,
        'superpoint': 256,
    }
    if net_model not in model_channels:
        raise ValueError(f"Model {net_model} not supported")
    return model_channels[net_model]


def seed_env() -> None:
    """
    Seeds the environment for reproducibility.

    - Sets seeds for PyTorch, NumPy, and the Python random module.
    - Ensures PyTorch's behavior is deterministic.

    Args:
        None

    Returns:
        None
    """
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    torch.backends.cudnn.deterministic = True


def init_device() -> torch.device:
    """
    Initializes the computation device.

    - Empties the CUDA cache.
    - Sets the default tensor type to CUDA if a GPU is available.
    - Prints whether the GPU or CPU is being used.

    Args:
        None

    Returns:
        torch.device: The device being used ('cuda' or 'cpu').
    """
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
        print_info("Using GPU")
    else:
        device = torch.device('cpu')
        print_warning("Using CPU")
    return device


def parse_args() -> mmengine.Config:
    """
    Parses and returns the command-line arguments and updates the configuration.

    The function defines the following arguments:
    - `--config` (required): The file path for the configuration file.
    - `--net_model` (optional): The network model type, options include 'alike-(l,n,s,t)' and 'superpoint'.
    - `--visualize` (optional): Whether to use visualization tools (default: False).

    The configuration file is loaded and updated with the parsed arguments:
    - Sets the experiment name (`expname`) based on the config.
    - Adds `net_model` and `visualize` options to the config.
    - Creates a `root_dir` in the config based on `basedir`, `expname`, and `net_model`.

    Returns:
        mmengine.Config: The updated configuration as a namespace object.

    Raises:
        SystemExit: If a required argument is missing, prints the help message and exits.
    """
    # ------------------- Read the args -------------------
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--config', required=True,
        help='Config file path (required)'
    )
    parser.add_argument(
        '--vox_side', type=int, required=True,
        help='Voxel side length N, for NxNxN'
    )
    parser.add_argument(
        '--net_model', type=str,
        help='Net model, choose between alike-(l,n,s,t), superpoint'
    )
    parser.add_argument(
        '--visualize', type=bool, default=False,
        help='Use visualization tools (default: False)'
    )
    # Try parsing arguments and handle missing mandatory arguments
    try:
        parsed_args = parser.parse_args()
    except SystemExit:
        print("\nError: Missing required arguments.\n")
        parser.print_help()
        raise

    # ------------------- Load and update configuration -------------------
    cfg = mmengine.Config.fromfile(parsed_args.config)

    cfg.model_and_render.num_voxels = parsed_args.vox_side ** 3

    if parsed_args.net_model != 'None':
        cfg.data.net_model = parsed_args.net_model

    # Update configuration with parsed arguments
    cfg['expname'] = f"{cfg['expname']}"
    cfg['net_model'] = parsed_args.net_model
    cfg['visualize'] = parsed_args.visualize

    # Create and set the root directory in the configuration
    root_dir = os.path.join(cfg.basedir, cfg.expname)
    root_dir = os.path.join(root_dir, cfg.net_model)
    cfg['root_dir'] = root_dir

    print_info(f"Creating project folder: {root_dir}")
    os.makedirs(root_dir, exist_ok=True)

    return cfg


def create_dataloader(dataset_type: str, data_path: str, scene: str) -> Dataloader:
    """
    Creates a dataloader object based on the dataset type.

    Args:
        dataset_type (str): The type of dataset to load.

    Returns:
        Dataloader: The dataloader object for the specified dataset type.
    """
    if dataset_type.lower() == '7scenes':
        return SevScenesDataloader(data_path, scene)
    elif dataset_type.lower() == 'cambridge':
        return CambridgeDataloader(data_path, scene)
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")


def create_tracker(net_model: str, K: np.ndarray, patch_size_half: int, path: str, distortion: np.ndarray = None,
                   log=False) -> BaseTracker:
    """
    Creates a tracker object based on the network model type.

    Args:
        net_model (str): The network model type.
        K (np.ndarray): The camera intrinsic matrix.
        patch_size_half (int): Half the size of the patch.
        path (str): The path to save the tracker data.
        distortion (np.ndarray): The camera distortion coefficients.

    Returns:
        Tracker: The tracker object for the specified network model.
    """
    if "alike" in net_model:
        return AlikeTracker(K=K, patch_size_half=patch_size_half, log=log, path=path, model=net_model,
                            distortion=distortion, lkt=False)
    elif "superpoint" in net_model:
        return SuperPointTracker(K=K, patch_size_half=patch_size_half, log=log, path=path,
                                 model='superpoint_v6_from_tf', distortion=distortion)
    else:
        raise ValueError("No tracker for this model")


def redirect2log(root_dir: str, name: str) -> Tuple[object, object]:
    """
    Redirects the standard output to a text file.

    Args:
        root_dir (str): The root directory to save the log file.

    Returns:
        Tuple[object, object]: The file object and the original stdout object.
    """
    print_warning("WRITING EVERYTHING TO TXT FILE")
    time_in_seconds = int(time.time())
    f = open(f'{root_dir}/{name}_{time_in_seconds}.txt', 'w')
    original_stdout = sys.stdout
    sys.stdout = f

    return f, original_stdout


def print_stats(name: str, data: np.array):
    """Print statistics for a given dataset.

    Args:
        name (str): The name of the dataset (e.g., "PSNR", "Time").
        data (np.ndarray): The dataset array.
    """
    print_info(f"----------------{name}----------------")
    print_info(f"Average {name}: {np.mean(data)}")
    print_info(f"Max {name}: {np.max(data)}")
    print_info(f"Min {name}: {np.min(data)}")
    print_info(f"Median {name}: {np.median(data)}")
    print_info(f"Std {name}: {np.std(data)}")


def load_model(base_path, model_class, top_n=None) -> torch.nn.Module:
    """
    Load a model from a checkpoint.

    Args:
        base_path (str): Path to the base directory containing the model checkpoints.
        model_class (class): The model class to initialize.
        top_n (int, optional): Number of top elements to retain in the model, if applicable.

    Returns:
        model (object): Loaded model instance or None if no checkpoint is found.
    """

    def load_checkpoint(path):
        if os.path.isfile(path):
            print_success(f"Checkpoint found at {path}.")
            return torch.load(path)
        return None

    def create_model_from_checkpoint(checkpoint):
        model = model_class(**checkpoint['model_kwargs'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print_success("Model loaded successfully!")
        return model

    checkpoint_path = os.path.join(base_path, 'model_ckpts/')
    checkpoint_file = os.path.join(checkpoint_path, 'model_last.tar')

    # Attempt to load checkpoint
    checkpoint = load_checkpoint(checkpoint_file)
    if not checkpoint:
        print_warning("No valid checkpoint found. Initializing a new model.")
        return None

    # Initialize and load model
    model = create_model_from_checkpoint(checkpoint)

    # Optional: Handle `top_n` if specified
    if top_n is not None:
        print_info(f"Configuring model for top_n={top_n}.")
        model.top_n(top_n)

    return model


def resume_model(base_path: str) -> FaVoRmodel:
    """
    Resumes a model from the most recent checkpoint.

    Args:
        base_path (str): Path to the base directory containing model checkpoints.

    Returns:
        FaVoRmodel: The resumed model instance or None if no checkpoint is found.
    """

    checkpoint_path = os.path.join(base_path, 'model_ckpts')
    checkpoint_pattern = "model_*.tar"
    checkpoint_list = glob.glob(os.path.join(checkpoint_path, checkpoint_pattern))

    if not checkpoint_list:
        print_error("No checkpoint found. Initializing a new model.")
        return None

    # Sort checkpoints by modification time to find the most recent
    checkpoint_last = max(checkpoint_list, key=os.path.getmtime)

    checkpoint = torch.load(checkpoint_last)
    print_success("Model resumed successfully!")
    print_success(f"Loading...")

    model = FaVoRmodel(**checkpoint['model_kwargs'])
    model.load_state_dict(checkpoint['model_state_dict'])
    print_info(f"Model loaded from {checkpoint_last}")

    return model


def store_model(model: torch.nn.Module, base_path: str, name: str) -> None:
    """
    Stores the model's voxels and network state to a checkpoint file.

    Args:
        model (object): The model to be stored.
        base_path (str): Path to the base directory where the model will be saved.
        name (str): The name of the checkpoint file.

    Returns:
        None
    """

    checkpoint_path = os.path.join(base_path, 'model_ckpts')

    # Create the checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_path, exist_ok=True)

    # Save the model's kwargs and state_dict to a checkpoint file
    checkpoint_file = os.path.join(checkpoint_path, f'{name}.tar')
    torch.save({
        'model_kwargs': model.get_kwargs(),
        'model_state_dict': model.state_dict(),
    }, checkpoint_file)

    print_success("Model stored successfully!")


def create_new_model(cfg_model, voxels_args: List, channels: int,
                     device: torch.device) -> FaVoRmodel:
    """
    Creates a new model instance based on the provided configuration and class.

    Args:
        cfg_model (dict): The configuration dictionary for the model.
        voxels_args (dict): Arguments to pass for creating voxels in the model.
        channels (int): Number of channels for the model.
        device (torch.device): The device to which the model will be transferred.

    Returns:
        FaVoRmodel: The created model instance.
    """

    # Prepare model kwargs by copying configuration and removing 'num_voxels'
    model_kwargs = {key: value for key, value in cfg_model.items() if key != 'num_voxels'}

    # Initialize the model with the provided arguments
    model = FaVoRmodel(
        voxels_args=voxels_args,
        channels=channels,
        **model_kwargs
    )

    # Transfer the model to the specified device
    model = model.to(device)

    return model


def create_optimizer_or_freeze_model(model, cfg_train, global_step):
    """
    Create an optimizer or freeze model parameters based on the learning rate configuration and decay.

    Args:
        model (torch.nn.Module): The model containing the parameters.
        cfg_train (Namespace): Training configuration with learning rate fields and decay settings.
        global_step (int): The current training step to calculate decay.

    Returns:
        MaskedAdam: An optimizer configured with the appropriate parameter groups.
    """
    decay_steps = cfg_train.lrate_decay * 1000
    decay_factor = 0.1 ** (global_step / decay_steps)

    param_groups = []

    first_param = False
    for key in cfg_train.keys():
        # Check for keys prefixed with 'lrate_'
        if not key.startswith('lrate_'):
            continue

        # Strip the prefix to get the parameter name
        param_name = key[len('lrate_'):]

        # Check if the model has the corresponding parameter
        if not hasattr(model, param_name):
            continue

        param = getattr(model, param_name)
        if param is None:
            continue  # Skip if the parameter does not exist

        if not first_param:
            first_param = True
            assert key != 'lrate_density' "First parameter must be density"

        # Compute the learning rate for the parameter
        lr = getattr(cfg_train, key) * decay_factor

        if lr > 0:
            # Add to the optimizer group if learning rate is positive
            if isinstance(param, torch.nn.Module):
                param = param.parameters()
            param_groups.append({
                'params': param,
                'lr': lr,
                'skip_zero_grad': (param_name in cfg_train.skip_zero_grad_fields)
            })
        else:
            # Freeze the parameter if learning rate is non-positive
            if hasattr(param, 'requires_grad'):
                param.requires_grad = False

    # Return the optimizer with the configured parameter groups
    return MaskedAdam(param_groups)


def log_early_stop(iterate, max_iter, estimated_dist_errors, estimated_angle_errors):
    """
    Handle early stop in the PnP iteration process.

    Args:
        iterate (int): Current iteration count.
        max_iter (int): Maximum number of iterations allowed.
        estimated_dist_errors (list): List storing distance errors per iteration.
        estimated_angle_errors (list): List storing angle errors per iteration.

    Logs warnings and fills remaining iterations with default error values (np.inf).
    """
    print_warning(f"Early stop at iteration {iterate}. No valid pose or matches found.")
    err_angle, err_dist = np.inf, np.inf
    for remaining_iter in range(iterate, max_iter):
        estimated_dist_errors[remaining_iter - 1].append(err_dist)
        estimated_angle_errors[remaining_iter - 1].append(err_angle)


def log_results(cfg, init_dist_errors, init_angle_errors, estimated_dist_errors, estimated_angle_errors, svfr_estimates,
                matches_per_iter, count_tests, dense_vlad=False):
    """
    Logs results for the pose estimation process, including errors and matches per iteration.

    Args:
        cfg: Configuration object with parameters for the process.
        init_dist_errors (list): Initial distance errors.
        init_angle_errors (list): Initial angle errors.
        estimated_dist_errors (list of lists): Estimated distance errors per iteration.
        estimated_angle_errors (list of lists): Estimated angle errors per iteration.
        svfr_estimates (list): Counts of valid poses per iteration.
        matches_per_iter (list): Number of matches per iteration.
        count_tests (int): Total number of test cases.
        dense_vlad (bool): Whether Dense VLAD is used.
    """
    print_info(
        "----------------WITH DENSE VLAD----------------" if dense_vlad else "----------------W/O DENSE VLAD----------------")

    # General setup
    print_info("---------------GENERAL SETUP-------------")
    print_info(f"Reprojection error: {cfg.data.reprojection_error[cfg.data.net_model]}")
    print_info(f"Feature matching threshold: {cfg.data.match_threshold[cfg.data.net_model]}")
    print_info(f"Voxel-grid size: {cfg.model_and_render.num_voxels} voxels")
    print_info("-----------------------------------------")

    # Initial errors
    print_info(f"Median initial distance error: {np.median(init_dist_errors):.4f} m")
    print_info(f"Median initial angle error: {np.median(init_angle_errors):.4f} deg")
    print_info(f"Max initial distance error: {np.max(init_dist_errors):.4f} m")
    print_info(f"Max initial angle error: {np.max(init_angle_errors):.4f} deg")
    print_info("-----------------------------------------")

    # Per-iteration results
    for i in range(len(svfr_estimates)):
        if svfr_estimates[i] > 0:
            print_info(f"{i + 1}st iter Median estimated distance error: {np.median(estimated_dist_errors[i]):.4f} m")
            print_info(f"{i + 1}st iter Median estimated angle error: {np.median(estimated_angle_errors[i]):.4f} deg")
            print_info(f"{i + 1}st iter Max estimated distance error: {np.max(estimated_dist_errors[i]):.4f} m")
            print_info(f"{i + 1}st iter Max estimated angle error: {np.max(estimated_angle_errors[i]):.4f} deg")
            ests_low = np.sum(
                np.logical_and(np.array(estimated_dist_errors[i]) < 0.05, np.array(estimated_angle_errors[i]) < 5.0)
            )
            print_info(f"{i + 1}st iter, <5 cm && <5 deg: {ests_low}/{count_tests}")
            print_info("-----------------------------------------")

    # Matches per iteration
    print_info("Number of matches per iteration:")
    for i in range(len(matches_per_iter)):
        if svfr_estimates[i] > 0:
            print_info(f"{i + 1}st iter: {matches_per_iter[i] / svfr_estimates[i]:.2f}")

    # Total iterations
    for i in range(len(svfr_estimates)):
        print_info(f"Total number of iterations {i + 1}: {svfr_estimates[i]}")

    print_info("-----------------------------------------")


def get_rays(K: torch.Tensor, c2w: torch.Tensor, pt: list = [0, 0], half_patch_size: int = 3) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """
    Computes the origin and direction of rays in world coordinates based on the camera intrinsics,
    camera-to-world transformation, and a pixel patch centered at a given point.

    Args:
        K (torch.Tensor): The camera intrinsic matrix (3x3), containing focal lengths and principal points.
        c2w (torch.Tensor): The camera-to-world transformation matrix (4x4), describing the camera's pose in the world frame.
        pt (list, optional): The center of the pixel patch as [x, y] coordinates in the image space. Defaults to [0, 0].
        half_patch_size (int, optional): Half the size of the square patch of pixels to process. Defaults to 3.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - rays_o (torch.Tensor): The ray origins in world coordinates with shape (patch_height, patch_width, 3).
            - rays_d (torch.Tensor): The ray directions in world coordinates with shape (patch_height, patch_width, 3).
    """
    i, j = torch.meshgrid(
        torch.linspace(-int(half_patch_size), int(half_patch_size), int(half_patch_size * 2 + 1), device=c2w.device),
        torch.linspace(-int(half_patch_size), int(half_patch_size), int(half_patch_size * 2 + 1), device=c2w.device),
        indexing='ij'  # pytorch's meshgrid has indexing='ij'
    )
    i = i.t().float()
    j = j.t().float()

    i, j = i + pt[0], j + pt[1]

    # Compute ray directions in the camera frame
    dirs = torch.stack([(i - K[0][2]) / K[0][0], (j - K[1][2]) / K[1][1], torch.ones_like(i)], -1)

    # Rotate ray directions from camera frame to world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  # dot product with rotation matrix

    # Compute ray origins in world coordinates
    rays_o = c2w[:3, 3].expand(rays_d.shape)

    return rays_o, rays_d


@torch.no_grad()
def get_training_rays(
        K: torch.Tensor,
        train_poses: torch.Tensor,
        pts: list,
        patch_size_half: int,
        device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """
    Computes ray origins and directions for training poses.

    Args:
        K (torch.Tensor): Camera intrinsic matrix of shape (3, 3).
        train_poses (torch.Tensor): Tensor of camera-to-world transformation matrices of shape (N, 4, 4), where N is the number of poses.
        pts (list): A list of 2D patch center coordinates (x, y) for each pose.
        patch_size_half (int): Half the size of the patch around each center point.
        device (torch.device): The device where the output tensors will be allocated.

    Returns:
        tuple[torch.Tensor, torch.Tensor, list]:
            - rays_o_tr (torch.Tensor): Ray origins of shape (N, patch_size, patch_size, 3).
            - rays_d_tr (torch.Tensor): Ray directions of shape (N, patch_size, patch_size, 3).
            - imsz (list): A list of integers indicating image sizes for each pose.
    """
    patch_size = patch_size_half * 2 + 1
    rays_o_tr = torch.zeros([len(train_poses), patch_size, patch_size, 3], device=device)
    rays_d_tr = torch.zeros([len(train_poses), patch_size, patch_size, 3], device=device)
    imsz = [1] * len(train_poses)  # Placeholder for image sizes

    for i, c2w in enumerate(train_poses):
        # Compute rays for the given pose and patch center
        rays_o, rays_d = get_rays(K, c2w, pt=pts[i], half_patch_size=patch_size_half)
        rays_o_tr[i].copy_(rays_o.to(device))
        rays_d_tr[i].copy_(rays_d.to(device))
        del rays_o, rays_d  # Free intermediate variables

    return rays_o_tr, rays_d_tr, imsz


def create_voxels_args(cfg_model, num_voxels: int, cfg_train, stage: str, tracks: list) -> list:
    """
    Creates a list of voxel arguments for scene representation reconstruction during training.

    Args:
        cfg_model: Configuration object for the model, containing parameters like `world_bound_scale`.
        num_voxels (int): The initial number of voxels for each track.
        cfg_train: Configuration object for training, containing parameters like `pg_scale`.
        stage (str): Current stage of the reconstruction process (e.g., "train", "validation").
        tracks (list): A list of track objects, each containing spatial bounds and attributes.

    Returns:
        list: A list of dictionaries, each containing voxel parameters:
            - `vox_id` (str): The ID of the track.
            - `xyz_min` (np.ndarray): The minimum coordinates of the voxel bounding box.
            - `xyz_max` (np.ndarray): The maximum coordinates of the voxel bounding box.
            - `num_voxels` (int): The adjusted number of voxels.
            - `point_w` (float): The weight or significance of the point in the track.
    """
    print(f'scene_rep_reconstruction ({stage}): train from scratch')

    voxels_args = []
    for i, track in enumerate(tracks):
        xyz_min, xyz_max = track.xyz_min, track.xyz_max

        # Collect voxel arguments for the current track
        voxel_args = {
            'vox_id': track.get_id(),
            'xyz_min': xyz_min,
            'xyz_max': xyz_max,
            'num_voxels': num_voxels,
            'point_w': track.point_w
        }
        voxels_args.append(voxel_args)

    return voxels_args


def unravel_index(index: int, shape: tuple) -> Tuple:
    """
    Converts a flat index into a tuple of indices corresponding to a given shape.

    Args:
        index (int): The flat index to be converted.
        shape (tuple): The shape of the multi-dimensional array.

    Returns:
        tuple: A tuple of indices representing the multi-dimensional index.
    """
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def model_size(model) -> float:
    """
    Calculates the memory size of a PyTorch model, including its parameters, buffers,
    and any additional keyword arguments defined by a custom method.

    Args:
        model (torch.nn.Module): The PyTorch model whose size is to be calculated.

    Returns:
        float: Total size of the model in megabytes (MB).
    """
    # Initialize size counters
    param_size = 0  # Size of model parameters
    buffer_size = 0  # Size of model buffers
    kwargs_size = 0  # Size of additional data in custom keyword arguments

    # Calculate the size of model parameters
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    # Calculate the size of model buffers
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    # Calculate the size of additional keyword arguments if the method exists
    if hasattr(model, 'get_kwargs'):
        for buffer_out in model.get_kwargs().values():
            for buffer_vox in buffer_out:
                for buffer in buffer_vox.values():
                    if isinstance(buffer, torch.Tensor):
                        kwargs_size += buffer.nelement() * buffer.element_size()
                    elif isinstance(buffer, np.ndarray):
                        kwargs_size += buffer.size * buffer.itemsize
                    else:
                        kwargs_size += sys.getsizeof(buffer)

    # Total size in bytes
    total_size = param_size + buffer_size + kwargs_size

    # Convert size to MB
    size_all_mb = total_size / (1024 ** 2)

    return size_all_mb
