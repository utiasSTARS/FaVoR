#  Copyright (c) 2023 Vincenzo Polizzi <vincenzo dot polizzi at mail dot utoronto dot ca>
#  (Space and Terrestrial Autonomous Robotic Systems Laboratory, University of Toronto,
#  Institute for Aerospace Studies).
#  This file is subject to the terms and conditions defined in the file
#  'LICENSE', which is part of this source code package.

from lib.models.favor_model import FaVoRmodel
from lib.utils_favor.misc_utils import print_info, model_size, seed_env, \
    init_device, parse_args, create_dataloader, create_tracker, load_model_hf
from lib.visualizer.o3dview import Favoro3d
from lib.visualizer.renderer import FavorRender
from lib.visualizer.widgets import FavorWidgets
import open3d.visualization.gui as gui

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

    # load the model
    model = load_model_hf(cfg.root_dir, FaVoRmodel, cfg.data.dataset_type, cfg.data.scene, cfg.data.net_model)

    # max 2000 voxels
    model.top_n(2000)

    print('model size: {:.3f} MB'.format(model_size(model)))

    if model is None:
        raise Exception("Model not loaded, train a model first!")

    gui.Application.instance.initialize()
    # create  new thread for the GUI
    o3dwin = Favoro3d(dataloader.camera.K, dataloader.camera.height, dataloader.camera.width)
    _, gt_pose, prior_pose = dataloader.get_test_line_at(0)

    o3dwin.gt_camera_pose_representation(gt_pose)
    o3dwin.prior_camera_pose_representation(prior_pose)

    o3dwin.create_voxel(model.get_bboxes())

    renderer = FavorRender(cfg=cfg, tracker=tracker, model=model, dataloader=dataloader)
    renderer.update_image()

    w = FavorWidgets(cfg=cfg, o3dwin=o3dwin, renderer=renderer)

    o3dwin.run()

    print_info('Done')
