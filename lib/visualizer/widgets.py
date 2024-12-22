from open3d.visualization import gui, rendering

from lib.visualizer.o3dview import Favoro3d
from lib.visualizer.renderer_new import FavorRender


class FavorWidgets:

    def __init__(self, cfg, o3dwin: Favoro3d, renderer: FavorRender):
        self.o3dwin = o3dwin
        self.renderer = renderer
        self.renderer.current_image()

        self.window = gui.Application.instance.create_window("FaVoR Widgets", 400, 350)

        w = self.window  # for more concise code

        em = w.theme.font_size

        layout = gui.Vert(2, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em,
                                         0.5 * em))

        # Match threshold slider
        match_slider = gui.Slider(gui.Slider.DOUBLE)
        match_slider.set_limits(0, 1)
        match_slider.double_value = cfg.data.match_threshold[cfg.data.net_model]
        match_slider.set_on_value_changed(lambda value: self.change_match_thr(value))

        # Reprojection error slider
        repr_error_slider = gui.Slider(gui.Slider.DOUBLE)
        repr_error_slider.set_limits(0, 20)
        repr_error_slider.double_value = cfg.data.reprojection_error[cfg.data.net_model]
        repr_error_slider.set_on_value_changed(lambda value: self.change_repr_error_thr(value))

        # Solve localization button
        loc_button = gui.Button("Localize")
        loc_button.set_on_clicked(self.localize_image)

        # Localize current view button
        loc_current_button = gui.Button("Localize Current View")
        loc_current_button.set_on_clicked(self.localize_view)

        # Rotate button
        self.rotate_button = gui.Button("Rotate")
        self.rotate_button.set_on_clicked(self.rotate_view)

        # Next image button
        next_image_button = gui.Button(">")
        next_image_button.set_on_clicked(self.next_image)

        # Previous image button
        prev_image_button = gui.Button("<")
        prev_image_button.set_on_clicked(self.prev_image)

        # Horizontal layout for navigation buttons
        button_row = gui.Horiz(5)
        button_row.add_child(prev_image_button)
        button_row.add_child(next_image_button)

        # Add components to the panel
        layout.add_child(gui.Label("Match Threshold"))
        layout.add_child(match_slider)
        layout.add_child(gui.Label("Reprojection Error"))
        layout.add_child(repr_error_slider)
        layout.add_child(gui.Label("Solve localization"))
        layout.add_child(loc_button)
        layout.add_child(loc_current_button)
        layout.add_child(self.rotate_button)
        layout.add_child(gui.Label("Navigate images"))
        layout.add_child(button_row)

        # Quit button. (Typically this is a menu item)
        button_layout = gui.Horiz()
        ok_button = gui.Button("Quit")
        ok_button.set_on_clicked(self.on_quit)
        button_layout.add_stretch()
        button_layout.add_child(ok_button)
        layout.add_child(button_layout)

        # We're done, set the window's layout
        w.add_child(layout)

    # Define methods that might be used for handling events
    def on_quit(self):
        print("Quit")
        self.o3dwin.stop()
        self.window.stop()
        gui.Application.instance.quit()

    def change_match_thr(self, value):
        # Placeholder for match threshold change logic
        print(f"Match threshold set to {value}")
        self.renderer.iter_pnp.match_threshold = value

    def change_repr_error_thr(self, value):
        # Placeholder for reprojection error threshold change logic
        print(f"Reprojection error set to {value}")
        self.renderer.iter_pnp.reprojection_error = value

    def localize_image(self):
        # Placeholder for image localization logic
        print("Localizing image...")
        pose, landmarks = self.renderer.localize_image()
        if pose is not None:
            self.o3dwin.est_camera_pose_representation(self.renderer.current_gt_pose, pose)
            self.o3dwin.rays_representation(landmarks, pose)

    def localize_view(self):
        # Placeholder for current view localization logic
        print("Localizing current view...")

    def rotate_view(self):
        # Placeholder for view rotation logic
        print("Rotating view...")
        self.o3dwin.rotate()

    def next_image(self):
        # Placeholder for next image logic
        print("Next image...")
        current_gt_pose, current_prior_pose = self.renderer.next_image()

        self.o3dwin.gt_camera_pose_representation(current_gt_pose)
        self.o3dwin.prior_camera_pose_representation(current_prior_pose)

    def prev_image(self):
        # Placeholder for previous image logic
        print("Previous image...")
        current_gt_pose, current_prior_pose = self.renderer.previous_image()

        self.o3dwin.gt_camera_pose_representation(current_gt_pose)
        self.o3dwin.prior_camera_pose_representation(current_prior_pose)
