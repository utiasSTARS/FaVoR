from copy import deepcopy

import numpy as np
import torch
from overrides import override

from lib.feature_extractors.base_extractor import BaseExtractor
from thirdparty.ALIKE.alike import ALike


class AlikeExtractor(BaseExtractor, ALike):
    def __init__(self,
                 # ================================== feature encoder
                 c1: int = 32, c2: int = 64, c3: int = 128, c4: int = 128, dim: int = 128,
                 single_head: bool = False,
                 # ================================== detect parameters
                 radius: int = 2,
                 top_k: int = 500, scores_th: float = 0.5,
                 n_limit: int = 5000,
                 device: str = 'cuda:0',
                 model_path: str = ''
                 ):
        BaseExtractor.__init__(self)  # init class BaseExtractor
        ALike.__init__(self,
                       c1, c2, c3, c4, dim, single_head,
                       radius, top_k, scores_th, n_limit,
                       device, model_path)  # init class Alike

    @override
    def forward(self, x) -> dict:
        H, W, three = x.shape
        assert three == 3, "input image shape should be [HxWx3]"

        # ==================== image size constraint
        image = deepcopy(x)

        # ==================== convert image to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(self.device).to(torch.float32).permute(2, 0, 1)[None]

        if image.max() > 1.0:
            image = image.to(torch.float32) / 255.

        # ==================== extract keypoints

        with torch.no_grad():
            descriptor_map, scores_map = self.extract_dense_map(image)
            keypoints, descriptors, scores, _ = self.dkd(scores_map, descriptor_map,
                                                         sub_pixel=True)
            keypoints, descriptors, scores = keypoints[0], descriptors[0], scores[0]
            keypoints = (keypoints + 1) / 2 * keypoints.new_tensor([[W - 1, H - 1]])

        indices = torch.argsort(scores, descending=True)
        keypoints = keypoints[indices]
        descriptors = descriptors[indices]

        return {'keypoints': keypoints.cpu().numpy(),
                'descriptors': descriptors.cpu().numpy(),
                'scores_map': scores_map.cpu().numpy()[0],
                'feature_map': descriptor_map.cpu().numpy()[0]}
