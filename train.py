import atexit
import sys
import time

import mmengine
from torch import nn
from torch.nn import CosineSimilarity
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F

from lib import utils
from lib.models import SVFRmodel

from lib.utils_svfr.svfr_utils import get_training_rays, create_voxels_args, load_model, resume_model, \
    create_new_model, store_model, seed_env, init_device, parse_args, create_dataloader, create_tracker, redirect2log, \
    print_stats, model2channels
from lib.utils_svfr.log_utils import print_error, print_info, print_success

import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import CosineSimilarity
import numpy as np
import atexit


def train(model: SVFRmodel, cfg: mmengine.Config, K: np.ndarray, device: torch.device, tracks, map_track,
          channels: int):
    """Train the SVFR model on voxel features."""
    # Setup parameters
    max_points = 1500 if cfg.data.dataset_type.lower() == '7scenes' else 10000
    max_voxels = min(max_points, len(tracks))
    count_success_voxels = 0

    # Register model saving on exit
    atexit.register(
        lambda: store_model(model, cfg.root_dir, 'model_partial') if count_success_voxels > 0 else print_info(
            "No voxels trained"))

    # Statistics tracking
    all_psnrs, all_times = [], []
    cos = CosineSimilarity(dim=1, eps=1e-6)

    # Training loop
    for v_id, vox in (bar := tqdm(enumerate(model.voxels), total=len(model.voxels))):
        if vox.trained:
            continue

        time_start = time.time()
        optimizer = setup_voxel_training(vox, cfg)
        track = get_voxel_track(vox, tracks, map_track)

        if track is None or not validate_track(vox, track):
            continue

        feature_tr, rays_o_tr, rays_d_tr, imsz = prepare_training_data(track, K, device)
        cnt = count_voxel_views(vox, rays_o_tr, rays_d_tr, imsz, cfg)

        if cnt is None or not setup_voxel_mask(vox, optimizer, cnt):
            continue

        # Train the voxel
        psnr = train_voxel(vox, feature_tr, rays_o_tr, rays_d_tr, channels, optimizer, cfg, cos)

        delta_time = time.time() - time_start
        count_success_voxels += process_training_result(vox, psnr, delta_time, all_psnrs, all_times, track)

        bar.set_description(f"count voxels: {count_success_voxels}, psnr: {np.mean(all_psnrs):.4f}")

        if count_success_voxels >= max_voxels:
            break

    finalize_training(model, all_psnrs, all_times, cfg)


def setup_voxel_training(vox, cfg):
    """Prepare voxel for training."""
    vox.is_training()
    return utils.create_optimizer_or_freeze_model(vox, cfg.coarse_train, global_step=0)


def get_voxel_track(vox, tracks, map_track):
    """Retrieve the track for a given voxel."""
    track_id = map_track.get(f'{vox.vox_id}')
    return tracks[track_id] if track_id else None


def validate_track(vox, track):
    """Ensure the voxel and track points match."""
    if (track.point_w != vox.point_w).all():
        print_error(f"Point mismatch: {track.point_w} != {vox.point_w}")
        vox.trained = False
        return False
    return True


def prepare_training_data(track, K, device):
    """Prepare features and rays for training."""
    feature_tr = torch.tensor(np.stack(track.features, axis=0), dtype=torch.float32, device=device)
    rays_o_tr, rays_d_tr, imsz = get_training_rays(K=K,
                                                   train_poses=track.get_poses_tensor(device),
                                                   pts=track.get_pts_tensor(device),
                                                   patch_size_half=track.patch_size_half,
                                                   device=device)
    return feature_tr, rays_o_tr, rays_d_tr, imsz


def count_voxel_views(vox, rays_o_tr, rays_d_tr, imsz, cfg):
    """Compute the number of views for a voxel."""
    try:
        return vox.voxel_count_views(rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=0.2,
                                     stepsize=cfg.coarse_model_and_render.stepsize, downrate=1)
    except Exception as e:
        print_error(f"Error in voxel view count: {e}")
        vox.trained = False
        return None


def setup_voxel_mask(vox, optimizer, cnt):
    """Configure voxel mask and learning rate."""
    cnt.clamp_(0, 100)
    optimizer.set_pervoxel_lr(cnt)
    vox.mask_cache.mask[cnt.squeeze() <= 2] = False
    return True


def train_voxel(vox, feature_tr, rays_o_tr, rays_d_tr, channels, optimizer, cfg, cos):
    """Train a single voxel."""
    psnr = 0
    for iter in range(2000):
        loss, render_result = voxel_training_step(vox, feature_tr, rays_o_tr, rays_d_tr, channels, optimizer, cos, iter,
                                                  cfg)
        psnr += utils.mse2psnr(loss.detach() / 4.).cpu().numpy()  # Desc range is [-1, 1]

        if iter > 1500:
            apply_total_variation_loss(vox, cfg, len(rays_o_tr))

        optimizer.step()

    return psnr / 2000


def voxel_training_step(vox, feature_tr, rays_o_tr, rays_d_tr, channels, optimizer, cos, iter, cfg):
    """Perform a single training iteration for a voxel."""
    sel_b = torch.randint(feature_tr.shape[0], [1024])
    sel_r = torch.randint(feature_tr.shape[1], [1024])
    sel_c = torch.randint(feature_tr.shape[2], [1024])

    target = feature_tr[sel_b, sel_r, sel_c]
    rays_o = rays_o_tr[sel_b, sel_r, sel_c]
    rays_d = rays_d_tr[sel_b, sel_r, sel_c]

    render_result = vox(rays_o, rays_d)
    optimizer.zero_grad(set_to_none=True)

    loss = compute_loss(render_result, target, channels, cos, iter)
    loss.backward()
    return loss, render_result


def compute_loss(render_result, target, channels, cos, iter):
    """Calculate the training loss."""
    loss = F.mse_loss(render_result['desc'], target[..., :channels])

    pout = render_result['alphainv_last'].clamp(1e-6, 1 - 1e-6)
    entropy_loss = -(pout * torch.log(pout) + (1 - pout) * torch.log(1 - pout)).mean()
    loss += 0.1 * entropy_loss

    if iter > 1:
        cos_loss = 1. - cos(render_result['desc'], target[..., :channels])
        loss += 0.2 * cos_loss.mean()

    return loss


def apply_total_variation_loss(vox, cfg, ray_count):
    """Apply total variation loss to the voxel."""
    if cfg.fine_train.weight_tv_density > 0:
        vox.density_total_variation_add_grad(cfg.fine_train.weight_tv_density / ray_count, True)
    if cfg.fine_train.weight_tv_k0 > 0:
        vox.k0_total_variation_add_grad(cfg.fine_train.weight_tv_k0 / ray_count, True)


def process_training_result(vox, psnr, delta_time, all_psnrs, all_times, track):
    """Handle the result of voxel training."""
    if psnr < 20.:
        vox.trained = False
    else:
        vox.trained = True
        vox.psnr = psnr
        vox.images_seen = track.get_frames_ids()
        all_psnrs.append(psnr)
        all_times.append(delta_time)
        return 1
    return 0


def finalize_training(model, all_psnrs, all_times, cfg):
    """Finalize training, remove untrained voxels, and save the model."""
    model.voxels = nn.ModuleList([vox for vox in model.voxels if vox.trained])
    print_stats("PSNR", np.array(all_psnrs))
    print_stats("Time", np.array(all_times))
    store_model(model, cfg.root_dir, 'model_last')
    print_success("Model trained and stored")


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
        raise Exception("Tracker is empty, run the tracker first")

    # retrieve the tracks and learn the points
    tracks = tracker.get_tracks(min_len=cfg.data.min_track_length, sort=True)

    print_info(f"Total tracks > {cfg.data.min_track_length}: {len(tracks)}")
    print_info(f"Min track length: {min([len(t) for t in tracks])}, max track len: {max([len(t) for t in tracks])}")

    channels = model2channels(cfg.data.net_model)
    print_info(f"\nChannels: {channels}")

    # Load model if exists
    model = load_model(cfg.root_dir, SVFRmodel)

    # train
    if not (model is None):
        print_success("Model already exists, skipping training...")
        exit(0)

    # create a log file and redirect stdout there
    f, original_stdout = redirect2log(cfg.root_dir)

    model = resume_model(cfg.root_dir, SVFRmodel)
    if model is None:
        print_info("Model not loaded, creating a new one...")
        voxels_args = create_voxels_args(cfg_model=cfg.coarse_model_and_render,
                                         num_voxels=cfg.coarse_model_and_render.num_voxels,
                                         cfg_train=cfg.coarse_train,
                                         stage='coarse',
                                         tracks=tracks)

        model = create_new_model(cfg_model=cfg.coarse_model_and_render,
                                 model_class=SVFRmodel,
                                 voxels_args=voxels_args,
                                 channels=channels,
                                 device=device)
    else:
        print_info("Model loaded, continuing training...")

    map_track = {}
    for i, t in enumerate(tracks):
        map_track[f'{t.get_id()}'] = i

    # start timing
    start_time = int(time.time())

    # Call the training function
    train(model, cfg, dataloader.camera.K, device, tracks, map_track, channels)

    print_info("Training done")

    # log time
    print_info(f"Training took {(time.time() - start_time)} seconds")

    sys.stdout = original_stdout
    print_info("Training done")
