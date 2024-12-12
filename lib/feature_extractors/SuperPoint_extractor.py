import os

import numpy as np
import torch
from overrides import override

from lib.feature_extractors.base_extractor import BaseExtractor
from thirdparty.SuperPoint.superpoint_pytorch import SuperPoint, batched_nms, sample_descriptors, select_top_k_keypoints


class SuperPointExtractor(BaseExtractor, SuperPoint):
    def __init__(self,
                 model_name: str = None,
                 device: str = 'cuda'):
        BaseExtractor.__init__(self)  # init class BaseExtractor
        SuperPoint.__init__(self)  # init class SuperPoint

        self.device = device
        # load pretrained
        if model_name is not None:
            # model is inside thirdparty
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
            pretrained_path = os.path.join(project_root, 'thirdparty', 'SuperPoint', 'weights', f'{model_name}.pth')

            # Check if the path exists
            if os.path.exists(pretrained_path):
                print(f'loading {pretrained_path}')
                state_dict = torch.load(pretrained_path, 'cpu')
                self.load_state_dict(state_dict, strict=True)
                self.to(device)
                self.eval()
            else:
                raise FileNotFoundError(f'cannot find pretrained model: {pretrained_path}')

    @override
    def forward(self, x) -> dict:

        # ==================== convert image to tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device).to(torch.float32).permute(2, 0, 1)[None]

        if x.max() > 1.0:
            x = x.to(torch.float32) / 255.

        """Forward pass."""
        if x.shape[1] == 3:  # RGB to gray
            scale = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            x = (x * scale).sum(1, keepdim=True)
        image = x.to(self.device)
        features = self.backbone(image)
        descriptors_dense = torch.nn.functional.normalize(
            self.descriptor(features), p=2, dim=1
        )

        # Decode the detection scores
        scores = self.detector(features)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.stride, self.stride)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(
            b, h * self.stride, w * self.stride
        )
        scores = batched_nms(scores, self.conf.nms_radius)

        # Discard keypoints near the image borders
        if self.conf.remove_borders:
            pad = self.conf.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1

        # Extract keypoints
        if b > 1:
            idxs = torch.where(scores > self.conf.detection_threshold)
            mask = idxs[0] == torch.arange(b, device=scores.device)[:, None]
        else:  # Faster shortcut
            scores = scores.squeeze(0)
            idxs = torch.where(scores > self.conf.detection_threshold)

        # Convert (i, j) to (x, y)
        keypoints_all = torch.stack(idxs[-2:], dim=-1).flip(1).float()
        scores_all = scores[idxs]

        keypoints = []
        scores = []
        descriptors = []
        for i in range(b):
            if b > 1:
                k = keypoints_all[mask[i]]
                s = scores_all[mask[i]]
            else:
                k = keypoints_all
                s = scores_all
            if self.conf.max_num_keypoints is not None:
                k, s = select_top_k_keypoints(k, s, self.conf.max_num_keypoints)
            d = sample_descriptors(k[None], descriptors_dense[i, None], self.stride)
            keypoints.append(k)
            scores.append(s)
            descriptors.append(d.squeeze(0).transpose(0, 1))

        return {
            "keypoints": keypoints[0].detach().cpu().numpy(),
            "keypoint_scores": scores[0].detach().cpu().numpy(),
            "descriptors": descriptors[0].detach().cpu().numpy(),
            'feature_map': descriptors_dense[0].detach().cpu().numpy()
        }
