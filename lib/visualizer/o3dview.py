import copy

import cv2
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from lib.utils_favor.transform_utils import CV2O3D
from lib.utils_favor.visualizer_utils import to8b


class Favoro3d:

    def __init__(self, K, H, W):
        self.rotate_bool = False

        # Create Open3D visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=W, height=H)
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=.2, origin=[0, 0, 0]))

        self.vis.register_animation_callback(self.view_callback)

        # create camera frustum
        self.cam_frustum_points = self.create_camera_model(K[0, 0], H, W)

        # create gt camera
        self.cam_gt_frustum = o3d.geometry.LineSet()
        self.vis.add_geometry(self.cam_gt_frustum)

        # create prior camera
        self.cam_prior_frustum = o3d.geometry.LineSet()
        self.vis.add_geometry(self.cam_prior_frustum)

        # create estimated camera
        self.est_cam_ets_frustum = o3d.geometry.LineSet()
        self.vis.add_geometry(self.est_cam_ets_frustum)

        # line set
        self.line_set = o3d.geometry.LineSet()
        self.vis.add_geometry(self.line_set)

        self.landmarks_rays = o3d.geometry.LineSet()
        self.vis.add_geometry(self.landmarks_rays)

        self.pose = torch.eye(4, device='cuda', dtype=torch.float32)
        self.prev_pose = None

        self.current_view = None

    def view_callback(self, vis):
        ctr = vis.get_view_control()
        cam = copy.deepcopy(ctr.convert_to_pinhole_camera_parameters())
        self.pose = torch.tensor(copy.deepcopy(cam.extrinsic), device='cuda', dtype=torch.float32)

        if self.prev_pose is None or not torch.eq(self.prev_pose, self.pose).all():
            c_view = copy.deepcopy(
                cv2.cvtColor(np.asarray(vis.capture_screen_float_buffer(True)), cv2.COLOR_BGR2RGB))
            self.current_view = to8b(c_view)

        if self.rotate_bool:
            ctr.rotate(20.0, 0.0)

        # self.renderer.update_image()

    def get_view(self):
        if self.current_view is None:
            ctr = self.vis.get_view_control()
            cam = copy.deepcopy(ctr.convert_to_pinhole_camera_parameters())
            self.pose = torch.tensor(copy.deepcopy(cam.extrinsic), device='cuda', dtype=torch.float32)
            c_view = copy.deepcopy(
                cv2.cvtColor(np.asarray(self.vis.capture_screen_float_buffer(True)), cv2.COLOR_BGR2RGB))
            self.current_view = to8b(c_view)

        return self.current_view

    def run(self):
        # new thread for the visualization
        self.vis.run()

    def stop(self):
        self.vis.destroy_window()

    def create_camera_model(self, f, H, W):
        def px2m(px):
            return px * 0.0002645833

        focal_m = px2m(f)  # https://www.unitconverters.net/typography/pixel-x-to-meter.htm
        h = px2m(H) / 2
        w = px2m(W) / 2
        left_top = np.array([-w, h, focal_m])
        right_top = np.array([w, h, focal_m])
        left_bottom = np.array([-w, -h, focal_m])
        right_bottom = np.array([w, -h, focal_m])

        return np.array([
            [0.0, 0.0, 0.0],  # Camera center
            left_top,  # Left top
            right_top,  # Right top
            right_bottom,  # Right bottom
            left_bottom,  # Left bottom
        ])

    def gt_camera_pose_representation(self, pose):
        # Convert pose matrix to Open3D format and extract translation and rotation
        p = CV2O3D(pose)
        translation = p[:3, 3]
        rotation = p[:3, :3]

        # Compute the transformed camera frustum points
        cam_frustum_points_ = np.dot(self.cam_frustum_points, rotation.T) + translation

        # Set up the ground truth camera frustum lines and points
        self.cam_gt_frustum.colors = o3d.utility.Vector3dVector([[0, 0, 1]] * 8)  # Blue color for all points
        self.cam_gt_frustum.lines = o3d.utility.Vector2iVector([
            [0, 1], [0, 2], [0, 3], [0, 4],  # Connect center to corners
            [1, 2], [2, 3], [3, 4], [4, 1]  # Connect the corners
        ])
        self.cam_gt_frustum.points = o3d.utility.Vector3dVector(cam_frustum_points_)

        self.vis.update_geometry(self.cam_gt_frustum)
        self.vis.update_renderer()

    def prior_camera_pose_representation(self, pose):
        # Convert pose to numpy if it's a torch tensor
        p = pose.detach().cpu().numpy() if isinstance(pose, torch.Tensor) else pose

        # Extract translation and rotation from the pose matrix
        translation = p[:3, 3]
        rotation = p[:3, :3]

        # Compute transformed camera frustum points
        cam_frustum_points_ = (rotation @ np.array(self.cam_frustum_points).T + translation[:, None]).T

        # Set up the camera frustum lines and points
        self.cam_prior_frustum.colors = o3d.utility.Vector3dVector([[1, 0, 1]] * 8)  # Magenta color
        self.cam_prior_frustum.lines = o3d.utility.Vector2iVector([
            [0, 1], [0, 2], [0, 3], [0, 4],  # Connect center to corners
            [1, 2], [2, 3], [3, 4], [4, 1]  # Connect the corners
        ])
        self.cam_prior_frustum.points = o3d.utility.Vector3dVector(cam_frustum_points_)

        # Update the visualization
        self.vis.update_geometry(self.cam_prior_frustum)
        self.vis.update_renderer()

    def est_camera_pose_representation(self, curr_pose, pose):
        # Extract translation and rotation from the pose matrix
        p = CV2O3D(pose)
        translation = p[:3, 3]
        rotation = p[:3, :3]

        # Update the line set (representing the camera pose)
        self.line_set.points = o3d.utility.Vector3dVector([curr_pose[:3, 3], translation])
        self.line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
        self.line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # Green color
        self.vis.update_geometry(self.line_set)

        # Compute the camera frustum points and update the visualization
        cam_frustum_points_ = np.dot(self.cam_frustum_points, rotation.T) + translation

        # Set up the estimated camera frustum lines and points
        self.est_cam_ets_frustum.colors = o3d.utility.Vector3dVector([[0, 0, 0]] * 8)  # Black color
        self.est_cam_ets_frustum.lines = o3d.utility.Vector2iVector([
            [0, 1], [0, 2], [0, 3], [0, 4],  # Connect center to corners
            [1, 2], [2, 3], [3, 4], [4, 1]  # Connect the corners
        ])
        self.est_cam_ets_frustum.points = o3d.utility.Vector3dVector(cam_frustum_points_)

        # Update the frustum geometry and renderer
        self.vis.update_geometry(self.est_cam_ets_frustum)
        self.vis.update_renderer()

    def rays_representation(self, matched_landmarks, T_star):
        # Convert transformation matrix to Open3D format
        T_start_o3d = CV2O3D(T_star)
        translation = T_start_o3d[:3, 3]

        if len(matched_landmarks) > 0:
            # Initialize list to store rays
            rays = []

            for landmark in matched_landmarks:
                # Construct transformation matrix for the landmark
                T_pts_w = np.column_stack((np.eye(3), landmark))  # Identity rotation
                T_pts_w = np.row_stack((T_pts_w, [0, 0, 0, 1]))  # Add homogenous coordinate
                T_pts_w = CV2O3D(T_pts_w)

                # Append the ray from the landmark to the camera center
                rays.append([T_pts_w[:3, 3], translation])

            # Update line set geometry
            self.landmarks_rays.points = o3d.utility.Vector3dVector(np.array(rays).reshape(-1, 3))
            self.landmarks_rays.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(0, len(rays) - 1, 2)])
            self.landmarks_rays.colors = o3d.utility.Vector3dVector(
                [[1, 0.5, 0]] * len(self.landmarks_rays.lines))  # Orange color

            # Update visualization
            self.vis.update_geometry(self.landmarks_rays)
            self.vis.update_renderer()

    def rotate(self):
        self.rotate_bool = not self.rotate_bool

    def create_voxel(self, bboxes):
        aabb_01 = np.array([[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 1],
                            [0, 1, 0],
                            [1, 0, 0],
                            [1, 0, 1],
                            [1, 1, 1],
                            [1, 1, 0]])

        rot = np.diag([1, 1, 1])
        for box in tqdm(bboxes):
            xyz_min = box['xyz_min'].detach().cpu().numpy()
            xyz_max = box['xyz_max'].detach().cpu().numpy()

            # convert to blender
            T_pts_w = np.column_stack((rot, xyz_min))
            T_pts_w = np.row_stack((T_pts_w, [0, 0, 0, 1]))
            xyz_min = CV2O3D(T_pts_w)[:3, 3]

            T_pts_w = np.column_stack((rot, xyz_max))
            T_pts_w = np.row_stack((T_pts_w, [0, 0, 0, 1]))
            xyz_max = CV2O3D(T_pts_w)[:3, 3]

            out_bbox = o3d.geometry.LineSet()
            out_bbox.points = o3d.utility.Vector3dVector(xyz_min + aabb_01 * (xyz_max - xyz_min))
            out_bbox.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(12)])
            out_bbox.lines = o3d.utility.Vector2iVector(
                [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])
            self.vis.add_geometry(out_bbox)
