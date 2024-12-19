expname = None  # experiment name
basedir = './logs/'  # where to store ckpts and logs

''' Template of data options
'''
data = dict(
    scene=None,
    datadir=None,
    dataset_type=None,
    white_bkgd=None,
    patch_size_half=None,
    min_track_length=None,
    train_test_invert=None,
    train_rgb=None,
    reprojection_error={},
    match_threshold={},
    net_model=None
)

''' Template of training options
'''
train = dict(
    N_iters=2000,  # number of optimization steps
    N_rand=1024,  # batch size (number of random rays per optimization step)
    lrate_decay=10,  # lr decay by 0.1 after every lrate_decay*1000 steps
    lrate_density=1e-1,  # lr of density voxel grid
    lrate_k0=1e-2,  # lr of color/feature voxel grid
    pervoxel_lr=True,  # view-count-based lr
    weight_tv_density=0.1,  # weight of total variation loss of density voxel grid
    weight_tv_k0=0.1,  # weight of total variation loss of color/feature voxel grid
    skip_zero_grad_fields=[],  # the variable name to skip optimizing parameters w/ zero grad in each iteration
    # pg_scale=[1000, 2000, 3000, 4000],
    # skip_zero_grad_fields=['density', 'k0'],
)

''' Template of model and rendering options
'''
model_and_render = dict(
    num_voxels=3 * 3 * 3,  # expected number of voxel, default is 3x3x3
    density_config=dict(),
    k0_config=dict(),
    mask_cache_thres=1e-3,  # threshold to determine a tighten BBox in the fine stage
    alpha_init=1e-2,  # set the alpha values everywhere at the begin of training
    fast_color_thres=1e-7,  # threshold of alpha value to skip the fine stage sampled point
    world_bound_scale=1,  # rescale the BBox enclosing the scene
    stepsize=0.1,  # sampling stepsize in volume rendering
)
