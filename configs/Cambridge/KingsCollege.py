_base_ = '../default_base.py'

expname = 'KingsCollege'
basedir = './logs/Cambridge'

data = dict(
    scene='KingsCollege',
    datadir='./datasets/Cambridge/',
    dataset_type='Cambridge',
    white_bkgd=True,
    patch_size_half=3,
    min_track_length=5,
    train_test_invert=False,
    train_rgb=False,
    reprojection_error={'alike-n': 5., 'alike-s': 5., 'alike-t': 5., 'alike-l': 10., 'superpoint': 5.},
    match_threshold={'alike-n': .7, 'alike-s': .7, 'alike-t': .7, 'alike-l': .7, 'superpoint': .7},
    net_model='alike-n',
    n_of_voxels=10000
)
