import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


def main():
    app = gui.Application.instance
    app.initialize()

    # Create a window
    window = gui.Application.instance.create_window("Visualizer", 1024, 768)

    # Create a 3D widget for the cube
    widget3d = gui.SceneWidget()
    widget3d.scene = rendering.Open3DScene(window.renderer)
    window.add_child(widget3d)

    # Create a cube and add it to the 3D scene
    cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    cube.compute_vertex_normals()
    cube.translate((0, 0, 0))  # Center the cube at (0, 0, 0)
    material = rendering.MaterialRecord()
    material.shader = "defaultLit"  # Use a lit shader for better visuals
    widget3d.scene.add_geometry("Cube", cube, material)

    # Create an ImageWidget to display an image
    image_widget = gui.ImageWidget(o3d.io.read_image("/home/viciopoli/datasets/Cambridge/ShopFacade/seq1/frame00001.png"))
    window.add_child(image_widget)

    # Define the start and end points of the line
    image_center_3d = [-3, 0, 0]  # Assume the image is "placed" at (-3, 0, 0)
    cube_center = [0, 0, 0]

    # Create a line set to represent the line
    points = [image_center_3d, cube_center]
    lines = [[0, 1]]
    colors = [[1, 0, 0]]  # Red color for the line
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Add the line to the 3D scene
    widget3d.scene.add_geometry("Line", line_set, material)

    # Adjust the layout: Split window between 3D and Image widget
    def on_layout(layout_context):
        content_rect = window.content_rect
        widget3d.frame = gui.Rect(content_rect.x,
                                  content_rect.y,
                                  int(content_rect.width * 0.7),  # 70% of the width for 3D
                                  content_rect.height)
        image_widget.frame = gui.Rect(widget3d.frame.get_right(),
                                      content_rect.y,
                                      int(content_rect.width * 0.3),  # 30% of the width for image
                                      content_rect.height)

    window.set_on_layout(on_layout)
    app.run()


if __name__ == "__main__":
    main()

