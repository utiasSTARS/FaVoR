import os
import random, argparse
import threading

import cv2
import mmengine as mmcv
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from lib.models.favor_model import FaVoRmodel
from lib.trackers.ALIKE_Tracker import AlikeTracker
from lib.trackers.SuperPoint_Tracker import SuperPointTracker
from lib.utils_favor.misc_utils import load_model, print_info, print_error, model2channels, model_size, seed_env, \
    init_device, parse_args, create_dataloader, create_tracker, load_model_hf
from lib.visualizer.o3dview import Favoro3d
from lib.visualizer.open3dVisualizer import Visualizer
import torch
from lib.visualizer.renderer import Renderer
from lib.visualizer.renderer_new import FavorRender
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

# {
# 	"class_name" : "ViewTrajectory",
# 	"interval" : 29,
# 	"is_loop" : false,
# 	"trajectory" :
# 	[
# 		{
# 			"boundingbox_max" : [ 0.5458109974861145, 3.8454704284667969, 1.3089832634665073 ],
# 			"boundingbox_min" : [ -0.6610596776008606, -1.1417173147201538, -0.33474248647689819 ],
# 			"field_of_view" : 60.0,
# 			"front" : [ -0.023389452883518135, 0.94686700179279959, 0.32077377450428235 ],
# 			"lookat" : [ 0.0088655839743462962, 1.3914574842303504, 0.37513270769117851 ],
# 			"up" : [ 0.0048007432648002387, -0.32075147447774421, 0.94715122577361277 ],
# 			"zoom" : 0.2999999999999996
# 		}
# 	],
# 	"version_major" : 1,
# 	"version_minor" : 0
# }


# {
# 	"class_name" : "ViewTrajectory",
# 	"interval" : 29,
# 	"is_loop" : false,
# 	"trajectory" :
# 	[
# 		{
# 			"boundingbox_max" : [ 0.47701609134674072, 54.418228149414062, 13.856255531311035 ],
# 			"boundingbox_min" : [ -33.012702941894531, -29.080280303955078, -46.288570404052734 ],
# 			"field_of_view" : 60.0,
# 			"front" : [ -0.0016970563424922687, 0.074083820801354699, 0.99725057407616635 ],
# 			"lookat" : [ -15.80553805988642, 9.1765377718476628, -22.220228270116593 ],
# 			"up" : [ -0.99999158697484192, 0.003598451051258153, -0.0019690428052924294 ],
# 			"zoom" : 0.41999999999999971
# 		}
# 	],
# 	"version_major" : 1,
# 	"version_minor" : 0
# }
