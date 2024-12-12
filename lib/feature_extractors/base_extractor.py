#  Copyright (c) 2023 Vincenzo Polizzi <vincenzo dot polizzi at mail dot utoronto dot ca>
#  (Space and Terrestrial Autonomous Robotic Systems Laboratory, University of Toronto,
#  Institute for Aerospace Studies).
#  This file is subject to the terms and conditions defined in the file
#  'LICENSE', which is part of this source code package.

import torch.nn as nn


class BaseExtractor(nn.Module):
    """
    Base class for feature extractors.

    Feature extractors are used to extract keypoints and descriptors from input data.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x) -> dict:
        """Forward pass.

        Subclasses must implement this method to process input data.

        Args:
            x: input data (size: [batch_size, channels, height, width])
        Returns:
            dict: dictionary with output data
                - 'keypoints' (ndarray): Keypoints extracted from the input data.
                  Size: [num_keypoints, 2] where 2 is for x and y coordinates.
                - 'descriptors' (ndarray): Descriptors associated with keypoints.
                  Size: [num_keypoints, descriptor_dim]
                - 'scores_map' (ndarray): Confidence scores for each keypoint.
                  Size: [height, width]
                - 'feature_map' (ndarray): Feature map from the extractor.
                  Size: [channels, height, width]
        """
        raise NotImplementedError
