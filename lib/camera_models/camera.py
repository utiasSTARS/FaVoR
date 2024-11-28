import numpy as np


class Camera:
    def __init__(self, width, height, fx, fy, cx, cy, distortion=None):
        self.width = width
        self.height = height

        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy

        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])

        self.distortion = distortion

    def scale(self, scale):
        self.fx *= scale
        self.fy *= scale
        self.cx *= scale
        self.cy *= scale
        self.K = np.array([[self.fx, 0, self.cx],
                           [0, self.fy, self.cy],
                           [0, 0, 1]])
