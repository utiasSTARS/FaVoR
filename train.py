import atexit
import sys
import time

from torch import nn
from torch.nn import CosineSimilarity
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F

from lib import utils
from lib.models import SVFRmodel

from lib.utils_svfr.svfr_utils import print_info, print_success, \
    get_training_rays, create_voxels_args, load_model, resume_model, \
    create_new_model, store_model, seed_env, init_device, parse_args, create_dataloader, create_tracker, redirect2log, \
    print_stats
from lib.utils_svfr.log_utils import print_error
from lib.utils_svfr.visualizer_utils import test_visualizer


def train(model: SVFRmodel, cfg, K, device, tracks, map_track, channels):
    max_points = 1500 if cfg.data.dataset_type.lower() == '7scenes' else 10000
    max_voxels = min(max_points, len(tracks))

    count_success_voxels = 0
    atexit.register(
        lambda: store_model(model, cfg.root_dir, 'model_partial') if count_success_voxels > 0 else print_info(
            "No voxels trained"))

    all_psnrs = []
    all_times = []

    cos = CosineSimilarity(dim=1, eps=1e-6)
    for v_id, vox in (bar := tqdm(enumerate(model.voxels), total=len(model.voxels))):
        if vox.trained:
            continue

        time_start = time.time()
        vox.is_training()
        optimizer = utils.create_optimizer_or_freeze_model(vox, cfg.coarse_train, global_step=0)

        track = tracks[map_track[f'{vox.vox_id}']]

        if (track.point_w != vox.point_w).all():
            print_error(f"Point mismatch: {track.point_w} != {vox.point_w}")
            vox.trained = False
            continue

        feature_tr = torch.tensor(np.stack(track.features, axis=0),
                                  dtype=torch.float32,
                                  device=device)
        rays_o_tr, rays_d_tr, imsz = get_training_rays(K=K,
                                                       train_poses=track.get_poses_tensor(device),
                                                       pts=track.get_pts_tensor(device),
                                                       patch_size_half=track.patch_size_half,
                                                       device=device)

        try:
            cnt = vox.voxel_count_views(
                rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=0.2,
                stepsize=cfg.coarse_model_and_render.stepsize, downrate=1)
        except Exception as e:
            vox.trained = False
            continue
        # cnt.clamp_(0, 100)
        optimizer.set_pervoxel_lr(cnt)
        vox.mask_cache.mask[cnt.squeeze() <= 2] = False
        psnr = 0
        for iter in range(2000):
            sel_b = torch.randint(feature_tr.shape[0], [1024])
            sel_r = torch.randint(feature_tr.shape[1], [1024])
            sel_c = torch.randint(feature_tr.shape[2], [1024])

            target = feature_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]

            # volume rendering
            render_result = vox(rays_o, rays_d)

            optimizer.zero_grad(set_to_none=True)

            loss = F.mse_loss(render_result['desc'], target[..., :channels])

            psnr += utils.mse2psnr(loss.detach() / 4.)  # 4 is (1- (-1))^2 because the desc is in the range [-1, 1]

            pout = render_result['alphainv_last'].clamp(1e-6, 1 - 1e-6)
            entropy_last_loss = -(pout * torch.log(pout) + (1 - pout) * torch.log(1 - pout)).mean()
            loss += 0.1 * entropy_last_loss

            if iter > 1:
                # add sin loss
                cos_loss = 1. - cos(render_result['desc'], target[..., :channels])
                loss += 0.2 * cos_loss.mean()

            loss.backward()

            if iter > 1500:
                if cfg.fine_train.weight_tv_density > 0:
                    vox.density_total_variation_add_grad(
                        cfg.fine_train.weight_tv_density / len(rays_o), True)
                if cfg.fine_train.weight_tv_k0 > 0:
                    vox.k0_total_variation_add_grad(
                        cfg.fine_train.weight_tv_k0 / len(rays_o), True)

            decay_steps = cfg.coarse_train.lrate_decay * 1000
            decay_factor = 0.1 ** (1 / decay_steps)
            for i_opt_g, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = param_group['lr'] * decay_factor

            optimizer.step()

        delta_time = time.time() - time_start

        if psnr / (iter + 1) < 1.:  # was 30 before
            vox.trained = False
        else:
            psnr_val = (psnr / (iter + 1)).detach().cpu().numpy()
            all_psnrs.append(psnr_val)
            all_times.append(delta_time)
            vox.psnr = psnr_val
            vox.trained = True
            vox.images_seen = track.get_frames_ids()
            count_success_voxels += 1
            if count_success_voxels > max_voxels:
                # print_info(f"Max number of voxels reached: {MAX_VOXELS}")
                break
            # # Test
            if psnr / (iter + 1) > 30.:
                print(f"Voxel {vox.vox_id} trained in {iter} iterations, psnr: {psnr / (iter + 1)}")
                for test_rays_o, test_rays_d, feature_test in zip(rays_o_tr, rays_d_tr, feature_tr):
                    test_rays_o = test_rays_o.unsqueeze(0)
                    test_rays_d = test_rays_d.unsqueeze(0)
                    test_visualizer(vox, None, test_rays_o, test_rays_d, None, cfg.coarse_model_and_render.stepsize,
                                    feature_test,
                                    cfg.data.patch_size_half)

        bar.set_description(
            f"count_success_voxels: {count_success_voxels}, psnr: {np.mean(all_psnrs):.4f}")

    # remove voxels that are not trained
    model.voxels = nn.ModuleList([vox for vox in model.voxels if vox.trained])

    # some psnr stats
    print_stats("PSNR", np.array(all_psnrs))
    print_stats("Time", np.array(all_times))

    # store the model
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

    channels = tracks[0].feature_channels()
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

    train(model, cfg, dataloader.camera.K, device, tracks, map_track, channels)

    print_info("Training done")

    # log time
    print_info(f"Training took {(time.time() - start_time)} seconds")

    sys.stdout = original_stdout
    print_info("Training done")
