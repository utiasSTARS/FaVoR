#  Copyright (c) 2023 Vincenzo Polizzi <vincenzo dot polizzi at mail dot utoronto dot ca>
#  (Space and Terrestrial Autonomous Robotic Systems Laboratory, University of Toronto,
#  Institute for Aerospace Studies).
#  This file is subject to the terms and conditions defined in the file
#  'LICENSE', which is part of this source code package.

import copy
from typing import List, Tuple

import cv2
import numpy as np
import open3d as o3d
import torch
from sklearn.decomposition import PCA

from lib.utils_favor.transform_utils import B2O3D, CV2O3D, CV2B

mse2psnr = lambda x: -10. * torch.log10(x)
normalize = lambda x: (x - x.min()) / (x.max() - x.min())
to8b = lambda x: (255 * normalize(x)).astype(np.uint8)


def visualize_camera_poses_and_points(pose_lst: List[np.ndarray], landmarks: List[np.ndarray]) -> None:
    """
    Visualizes camera poses and 3D points using Open3D.

    Args:
        pose_lst (List[np.ndarray]): List of 4x4 pose matrices representing camera poses.
        landmarks (List[np.ndarray]): List of 3D landmarks to visualize.

    Returns:
        None: Displays the visualization in an Open3D window.
    """
    cam_frustum_lst = []
    translations = []

    for i, pose in enumerate(pose_lst):
        # Define the frustum points in the camera's local coordinate system
        cam_frustum_points = np.array([
            [0.0, 0.0, 0.0],  # Camera center
            [0.2, 0.2, 0.3],  # Top right corner
            [0.2, -0.2, 0.3],  # Bottom right corner
            [-0.2, -0.2, 0.3],  # Bottom left corner
            [-0.2, 0.2, 0.3]  # Top left corner
        ])

        # Extract translation and rotation from the pose matrix
        translation = pose[:3, 3]
        rotation = B2O3D(pose)[:3, :3]

        translations.append(copy.deepcopy(translation))

        # Transform frustum points to the world coordinate system
        cam_frustum_points = np.dot(cam_frustum_points, rotation.T) + translation

        # Create a LineSet for the camera frustum
        cam_frustum = o3d.geometry.LineSet()
        cam_frustum.points = o3d.utility.Vector3dVector(cam_frustum_points)
        cam_frustum.lines = o3d.utility.Vector2iVector([
            [0, 1], [0, 2], [0, 3], [0, 4],  # Camera center to corners
            [1, 2], [2, 3], [3, 4], [4, 1]  # Rectangle connecting corners
        ])

        # Assign color based on the index
        if i == 0:
            cam_frustum.colors = o3d.utility.Vector3dVector([[0, 0, 1]] * 8)  # Blue for the first camera
        elif i == 1:
            cam_frustum.colors = o3d.utility.Vector3dVector([[1, 0, 1]] * 8)  # Purple for the second camera
        else:
            cam_frustum.colors = o3d.utility.Vector3dVector([[0, 0, 0]] * 8)  # Black for others

        cam_frustum_lst.append(copy.deepcopy(cam_frustum))

    # Create LineSet connecting camera centers
    line_set = o3d.geometry.LineSet()
    if len(translations) > 1:
        line_set.points = o3d.utility.Vector3dVector(translations)
        line_set.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(translations) - 1)])
        line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * (len(translations) - 1))  # Green lines

    # Create a PointCloud for landmarks
    pcd = o3d.geometry.PointCloud()
    points = []
    if len(landmarks) > 0:
        for i, landmark in enumerate(landmarks):
            rot = np.diag([1, 1, 1])
            T_pts_w = np.column_stack((rot, landmark))
            T_pts_w = np.row_stack((T_pts_w, [0, 0, 0, 1]))
            T_pts_w = CV2O3D(T_pts_w)
            points.append(T_pts_w[:3, 3])

    pcd.points = o3d.utility.Vector3dVector(points)

    # Visualize using Open3D
    o3d.visualization.draw_geometries([
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]),
        *cam_frustum_lst,
        line_set,
        pcd
    ])


def jet_img(map: np.ndarray) -> np.ndarray:
    """
    Apply a jet colormap to an image and save it to a file.

    Args:
        map (np.ndarray): The input image or map to be processed.

    Returns:
        np.ndarray: The processed image if `return_img` is True; otherwise, None.

    Raises:
        ValueError: If the input map is not a valid image.
    """
    if map is None or not isinstance(map, np.ndarray):
        raise ValueError("Input map must be a valid numpy ndarray.")

    # Handle input with more than 3 channels
    if map.ndim > 2 and map.shape[-1] > 3:
        map = np.mean(map, axis=-1)

    # Normalize input values to 0-255 for colormap application
    normalized_img = cv2.normalize(map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply the jet colormap
    return cv2.applyColorMap(normalized_img, cv2.COLORMAP_JET)


def visualize_matches(image, match_img, matched_estimated_kpts, target_kpts, estimated_landmarks, pose, K):
    """
    Visualize matches during the PnP process, including estimated keypoints and ground truth.

    Args:
        image (np.ndarray): Original input image (H x W x C).
        match_img (np.ndarray): Image for visualization (H x 2W x C).
        matched_estimated_kpts (np.ndarray): Estimated matched keypoints (N x 2).
        target_kpts (np.ndarray): Target matched keypoints (N x 2).
        estimated_landmarks (np.ndarray): Estimated 3D landmarks (3 x N).
        pose (np.ndarray): Current camera pose (4 x 4).
        K (np.ndarray): Camera intrinsic matrix (3 x 3).

    Returns:
        np.ndarray: Updated visualization image.
    """
    if len(matched_estimated_kpts) == 0:
        return match_img

    # Draw matched estimated keypoints
    for kp in matched_estimated_kpts:
        match_img = cv2.circle(match_img, (int(kp[0]), int(kp[1])), 4, (0, 255, 255), 1)

    # Project estimated landmarks into 2D
    ppp = np.linalg.inv(pose)[:3, :]
    kpts_cam = ppp[:, :3] @ estimated_landmarks.transpose() + ppp[:, 3:]
    kps = K @ kpts_cam
    kps = kps[:2] / kps[2]
    kps = kps.transpose()
    for kp in kps:
        match_img = cv2.circle(match_img, (int(kp[0]), int(kp[1])), 2, (255, 0, 255), -1)

    # Draw target matches
    for est_kp in matched_estimated_kpts:
        match_img = cv2.circle(match_img, (int(image.shape[1] + est_kp[0]), int(est_kp[1])), 2, (0, 100, 255), -1)
    for matched_kp, targ_kp in zip(matched_estimated_kpts, target_kpts):
        match_img = cv2.circle(match_img, (int(matched_kp[0]), int(matched_kp[1])), 3, (255, 0, 255), -1)
        match_img = cv2.circle(match_img, (int(image.shape[1] + targ_kp[0]), int(targ_kp[1])), 3, (255, 0, 0), -1)

    # Add a legend to the visualization
    match_img = image_with_legend(
        match_img,
        colors=[(0, 255, 0), (0, 100, 255), (255, 0, 255), (255, 0, 0)],
        labels=['Ground Truth', 'All estimated', 'Matched estimated', 'Matched target']
    )

    # Display the visualization
    cv2.imshow('Matches difference', match_img)
    cv2.waitKey(1)

    return match_img


def score_to_rgb(value: float) -> Tuple[int, int, int]:
    """
    Maps a score (value) in the range [0, 1] to an RGB color representation.

    Args:
        value (float): The input score, expected to be in the range [0, 1].
                       Values outside this range will be clamped to [0, 1].

    Returns:
        Tuple[int, int, int]: The corresponding RGB color as a tuple (red, green, blue).
                              - Red decreases as the value approaches 1.
                              - Green increases as the value approaches 1.
                              - Blue remains 0.
    """
    # Clamp value to the range [0, 1]
    value = max(0, min(1, value))

    # Map value to RGB
    red = int((1 - value) * 255)
    green = int(value * 255)
    blue = 0

    return red, green, blue


def draw_kpts(frame, kps_prev, kps_curr):
    """
    Draws keypoint matches between two sets of points on a frame. Each match is visualized as a green line.

    Args:
        frame (Union[np.ndarray, torch.Tensor]): The input image/frame. Can be a NumPy array or a PyTorch tensor.
        kps_prev (Iterable[Tuple[float, float]]): Previous set of keypoints (e.g., from the first image).
        kps_curr (Iterable[Tuple[float, float]]): Current set of keypoints (e.g., from the second image).

    Returns:
        np.ndarray: The image with lines drawn between the corresponding keypoints.
    """
    # Check if the frame is a tensor, and convert it to a NumPy array if needed
    if isinstance(frame, torch.Tensor):
        frame_np = frame.detach().cpu().numpy()
        img_colored = to8b(frame_np.copy())  # Assuming to8b converts to 8-bit range [0, 255]
    else:
        img_colored = copy.deepcopy(frame)

    # Draw lines between corresponding keypoints
    for pt1, pt2 in zip(kps_prev, kps_curr):
        img_colored = cv2.line(
            img_colored,
            (int(pt1[0]), int(pt1[1])),  # Convert float to integer pixel coordinates
            (int(pt2[0]), int(pt2[1])),
            (0, 255, 0),  # Green color
            lineType=cv2.LINE_AA  # Anti-aliased line for smoother visualization
        )

    return img_colored


def visualize_images(imgs, r, c, quantize=False, wait_key=0):
    """
    Visualizes a grid of images in a single canvas.

    Args:
        imgs (List[np.ndarray]): List of images to display. Each image should be a NumPy array.
        r (int): Number of rows in the grid.
        c (int): Number of columns in the grid.
        quantize (bool, optional): Whether to quantize image values to 8-bit range if they are not already. Default is False.
        wait_key (int, optional): Time in milliseconds to wait for a key press. Default is 0 (wait indefinitely).

    Raises:
        ValueError: If the number of images does not match the grid size.

    Returns:
        None
    """
    if len(imgs) != r * c:
        raise ValueError("Number of images does not match the grid size")

    # Determine the maximum height, width, and channels across all images
    max_height = max(img.shape[0] for img in imgs)
    max_width = max(img.shape[1] for img in imgs)
    channels = imgs[0].shape[2] if imgs[0].ndim == 3 else 1

    # Create a blank canvas to hold all the images
    canvas = np.zeros((max_height * r, max_width * c, channels), dtype=np.uint8)

    # Fill the canvas with the images
    for idx, img in enumerate(imgs):
        row, col = divmod(idx, c)

        # Quantize the image to 8-bit if needed
        if quantize and np.max(img) <= 1.0:
            img = (img * 255).astype(np.uint8)

        # Convert grayscale images to 3 channels for uniformity
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Resize the image to fit the grid cell
        img_resized = cv2.resize(img, (max_width, max_height))

        # Determine the position on the canvas
        y_start, y_end = row * max_height, (row + 1) * max_height
        x_start, x_end = col * max_width, (col + 1) * max_width

        # Place the image on the canvas
        canvas[y_start:y_end, x_start:x_end, :] = img_resized

    # Display the final canvas
    cv2.namedWindow("Grid Visualization", cv2.WINDOW_NORMAL)
    cv2.imshow("Grid Visualization", canvas)
    cv2.waitKey(wait_key)
    cv2.destroyAllWindows()


######################################################################

class DescVisualizer:
    PCA_DIM = 3

    def __init__(self, descs: np.array):
        self.pca = PCA(n_components=self.PCA_DIM)

        if len(descs) > 0:
            self.fit(descs)

    def fit(self, descs):
        self.pca.fit(descs)

    def transform(self, descs):
        return self.pca.transform(descs)

    def render(self, imgs_descs, p):
        img = self.transform(imgs_descs)
        img = img.reshape(p, p, self.PCA_DIM)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-10)
        return img


def visualize_camera_poses(pose_lst_cv, bbox_lst_cuda=[], matched_landmarks=[]):
    import open3d as o3d

    # convert everything to blender and then to open3d
    pose_lst = [CV2B(p) for p in pose_lst_cv]

    # matched_landmarks = []
    # for landmark in matched_landmarks_cv:
    #     rot = np.diag([1, 1, 1])
    #     T_pts_w = np.column_stack((rot, landmark))
    #     T_pts_w = np.row_stack((T_pts_w, [0, 0, 0, 1]))
    #     T_pts_w = CV2O3D(T_pts_w)
    #     matched_landmarks.append(T_pts_w[:3, 3])

    cam_frustum_lst = []
    translations = []
    for i, pose in enumerate(pose_lst):
        cam_frustum_points = [
            [0.0, 0.0, 0.0],  # Camera center
            [0.2, 0.2, .30],  # Top right corner
            [0.2, -0.2, .30],  # Bottom right corner
            [-0.2, -0.2, .30],  # Bottom left corner
            [-0.2, 0.2, .30]  # Top left corner
        ]
        # Extract translation and rotation from the pose matrix
        translation = pose[:3, 3]
        rotation = B2O3D(pose)[:3, :3]

        # Flip the sign of the translation along the Z-axis to match Open3D's coordinate system
        # translation[2] *= -1
        translations.append(copy.deepcopy(translation))

        # Compute camera center in world coordinates
        # camera_center = -np.dot(rotation.T, translation)

        ##############################################
        # Transform camera frustum points to match the pose
        cam_frustum_points = np.dot(cam_frustum_points, rotation.T) + translation

        # Create camera frustum LineSet
        cam_frustum = o3d.geometry.LineSet()
        cam_frustum.points = o3d.utility.Vector3dVector(cam_frustum_points)

        if i == 0:
            cam_frustum.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(8)])
        elif i == 1:
            cam_frustum.colors = o3d.utility.Vector3dVector([[1, 0, 1] for _ in range(8)])
        else:
            cam_frustum.colors = o3d.utility.Vector3dVector([[0, 0, 0] for _ in range(8)])
        cam_frustum.lines = o3d.utility.Vector2iVector(
            [[0, 1], [0, 2], [0, 3], [0, 4],  # Connect center to corners
             [1, 2], [2, 3], [3, 4], [4, 1]])  # Connect corners to form a rectangle

        cam_frustum_lst.append(copy.deepcopy(cam_frustum))
        del cam_frustum, translation, rotation

    # Create LineSet connecting camera centers
    lines = []
    for i in range(len(pose_lst) - 2):
        lines.append([i, i + 1])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(translations[1:])
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(pose_lst) - 2)])

    # from landmarks to camera center
    landmarks_rays = []
    if len(matched_landmarks) > 0:
        for i, landmark in enumerate(matched_landmarks):
            rot = np.diag([1, 1, 1])
            T_pts_w = np.column_stack((rot, landmark))
            T_pts_w = np.row_stack((T_pts_w, [0, 0, 0, 1]))
            T_pts_w = CV2O3D(T_pts_w)

            land_frustum = o3d.geometry.LineSet()
            land_frustum.points = o3d.utility.Vector3dVector([T_pts_w[:3, 3], translations[-1]])
            land_frustum.lines = o3d.utility.Vector2iVector([[0, 1]])
            land_frustum.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(2)])
            landmarks_rays.append(land_frustum)

    aabb_01 = np.array([[0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 1],
                        [0, 1, 0],
                        [1, 0, 0],
                        [1, 0, 1],
                        [1, 1, 1],
                        [1, 1, 0]])
    bbox_lst = []
    rot = np.diag([1, 1, 1])
    for box in bbox_lst_cuda:
        xyz_min = box['xyz_min'].detach().cpu().numpy()
        xyz_max = box['xyz_max'].detach().cpu().numpy()

        # convert to blender
        T_pts_w = np.column_stack((rot, xyz_min))
        T_pts_w = np.row_stack((T_pts_w, [0, 0, 0, 1]))
        xyz_min = CV2O3D(T_pts_w)[:3, 3]
        T_pts_w = np.column_stack((rot, xyz_max))
        T_pts_w = np.row_stack((T_pts_w, [0, 0, 0, 1]))
        xyz_max = CV2O3D(T_pts_w)[:3, 3]

        # if (xyz_min > 3).any() or (xyz_max > 3).any(0):
        #     continue

        out_bbox = o3d.geometry.LineSet()
        out_bbox.points = o3d.utility.Vector3dVector(xyz_min + aabb_01 * (xyz_max - xyz_min))
        out_bbox.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(12)])
        out_bbox.lines = o3d.utility.Vector2iVector(
            [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])
        bbox_lst.append(out_bbox)

    # Show
    o3d.visualization.draw_geometries([
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]),
        *cam_frustum_lst, *bbox_lst, line_set, *landmarks_rays])


def visualize_bboxes(bboxes):
    aabb_01 = np.array([[0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 1],
                        [0, 1, 0],
                        [1, 0, 0],
                        [1, 0, 1],
                        [1, 1, 1],
                        [1, 1, 0]])
    bboxes_lst = []

    for bbox in bboxes:
        # Outer aabb
        out_bbox = o3d.geometry.LineSet()
        out_bbox.points = o3d.utility.Vector3dVector(
            bbox.xyz_min.cpu().numpy() + aabb_01 * (bbox.xyz_max.cpu() - bbox.xyz_min.cpu()).numpy())
        out_bbox.colors = o3d.utility.Vector3dVector([[1, 0, 0] for i in range(12)])
        out_bbox.lines = o3d.utility.Vector2iVector(
            [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])
        bboxes_lst.append(out_bbox)
    # Show
    o3d.visualization.draw_geometries([
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1, origin=[0, 0, 0]),
        *bboxes_lst])
    return bboxes_lst


def visualize_sampling_rays(rays_o, rays_d, tracks):
    # Create lists to hold points and edges
    all_points = []
    all_edges = []
    current_point_index = 0

    # Generate points and edges for each line
    for o, d in zip(rays_o, rays_d):
        # Generate points along the line
        t_values = np.linspace(-0.1, 1, num=10)  # Adjust the range and number of points as needed
        points = np.array([o + t * d for t in t_values])

        # Append points to the list of all points
        all_points.extend(points)

        # Generate edges for this line
        num_points = len(points)
        edges = [[current_point_index + i, current_point_index + i + 1] for i in range(num_points - 1)]
        all_edges.extend(edges)

        # Update the current point index
        current_point_index += num_points

    # Create a LineSet to visualize all lines
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(all_points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(all_edges))

    # Optionally, add colors for better visualization
    # colors = [[np.random.rand(), np.random.rand(), np.random.rand()] for _ in range(len(all_edges))]
    # line_set.colors = o3d.utility.Vector3dVector(colors)

    aabb_01 = np.array([[0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 1],
                        [0, 1, 0],
                        [1, 0, 0],
                        [1, 0, 1],
                        [1, 1, 1],
                        [1, 1, 0]])
    bboxes_lst = []
    for bbox in tracks:
        # Outer aabb
        out_bbox = o3d.geometry.LineSet()
        out_bbox.points = o3d.utility.Vector3dVector(
            bbox.xyz_min.cpu().numpy() + aabb_01 * (bbox.xyz_max.cpu() - bbox.xyz_min.cpu()).numpy())
        out_bbox.colors = o3d.utility.Vector3dVector([[1, 0, 0] for i in range(12)])
        out_bbox.lines = o3d.utility.Vector2iVector(
            [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])
        bboxes_lst.append(out_bbox)

    # Visualize the lines
    o3d.visualization.draw_geometries([line_set, *bboxes_lst])


def visualize_rays(rays, bbox, cams=None):
    pts = rays.detach().cpu().numpy()
    pts_len = len(pts)

    points = []
    c = 0
    a = range(0, pts_len, int(pts_len / 1000))
    d = pts[a]
    for p in d:
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.0005)
        mesh_sphere.paint_uniform_color([0.5, c, c])
        t = np.array([[1, 0, 0, p[0]], [0, 1, 0, p[1]], [0, 0, 1, p[2]], [0, 0, 0, 1]])
        mesh_sphere.transform(t)
        points.append(mesh_sphere)
        c += 0.001

    aabb_01 = np.array([[0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 1],
                        [0, 1, 0],
                        [1, 0, 0],
                        [1, 0, 1],
                        [1, 1, 1],
                        [1, 1, 0]])

    # Outer aabb
    out_bbox = o3d.geometry.LineSet()
    out_bbox.points = o3d.utility.Vector3dVector(
        bbox.xyz_min.cpu().numpy() + aabb_01 * (bbox.xyz_max.cpu() - bbox.xyz_min.cpu()).numpy())
    out_bbox.colors = o3d.utility.Vector3dVector([[1, 0, 0] for i in range(12)])
    out_bbox.lines = o3d.utility.Vector2iVector(
        [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])

    # Cameras
    cam_frustum_lst = []
    if cams is not None:
        for cam in cams:
            cam_frustum = o3d.geometry.LineSet()
            cam_frustum.points = o3d.utility.Vector3dVector(cam)
            if len(cam) == 5:
                cam_frustum.colors = o3d.utility.Vector3dVector([[0, 0, 0] for i in range(8)])
                cam_frustum.lines = o3d.utility.Vector2iVector(
                    [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 4], [4, 3], [3, 1]])
            elif len(cam) == 8:
                cam_frustum.colors = o3d.utility.Vector3dVector([[0, 0, 0] for i in range(12)])
                cam_frustum.lines = o3d.utility.Vector2iVector([
                    [0, 1], [1, 3], [3, 2], [2, 0],
                    [4, 5], [5, 7], [7, 6], [6, 4],
                    [0, 4], [1, 5], [3, 7], [2, 6],
                ])
            else:
                raise NotImplementedError
            cam_frustum_lst.append(cam_frustum)

    o3d.visualization.draw_geometries([
        # o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1, origin=[0, 0, 0]),
        *points, *cam_frustum_lst, out_bbox])


def render_voxels(renderer, pose, img, K, p=800, desc_extractor=None, matcher=None):
    kpts, target_kpts, result = None, None, None
    if desc_extractor is None:
        kpts, desc, render, depth = renderer.render_rgb(H=p, W=p, K=K, c2w=pose, depth=False)
        render = render.cpu().numpy()
        visualize_images([cv2.cvtColor(render, cv2.COLOR_BGR2RGB), img, np.abs(render - img)],
                         1,
                         2)

        kpts, target_kpts, result = matcher(kpts, desc, img, desc_extractor)
    else:
        kpts, desc, render, _ = renderer.render(H=p, W=p, K=K, c2w=pose, depth=False)
        render = render.reshape(p ** 2, render.shape[-1])
        render = render.cpu().numpy()
        descriptors_renderer = DescVisualizer(render)
        descriptors_renderer.render(render, p)

        kpts, target_kpts, result = matcher(kpts, desc, img, desc_extractor)

    return kpts, target_kpts, result


def image_with_legend(image, colors, labels):
    # Create a blank image for the legend
    text_offset_y = 30
    legend_height = 20 + len(colors) * text_offset_y
    legend_width = 200
    legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255

    # Write color labels onto the legend
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    for i, (color, label) in enumerate(zip(colors, labels)):
        legend = cv2.putText(legend, label, (30, text_offset_y * (i + 1)), font, font_scale, color, thickness)

    image[:legend_height, :legend_width] = legend
    return image
