_base_ = '../default_base.py'

expname = 'pumpkin'
basedir = './logs/7Scenes'

data = dict(
    scene='pumpkin',
    datadir='./datasets/7scenes/',
    dataset_type='7Scenes',
    white_bkgd=True,
    patch_size_half=3,
    min_track_length=5,
    train_test_invert=False,
    train_rgb=False,
    reprojection_error={'alike-n': 5., 'alike-s': 5., 'alike-t': 5., 'alike-l': 5., 'superpoint': 5.},
    match_threshold={'alike-n': .7, 'alike-s': .7, 'alike-t': .7, 'alike-l': .7, 'superpoint': .8},
    net_model='alike-n'  # 'superpoint_v6_from_tf'  # 'aliked-n32'  # 'aliked-t16' # 'aliked-n32'
)
