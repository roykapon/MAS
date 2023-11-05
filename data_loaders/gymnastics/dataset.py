import numpy as np
from data_loaders.base_dataset import AugDataset


class GymnasticsDataset(AugDataset):
    def __init__(self, datapath, confidence_threshold, distance_threshold, *args, **kwargs):
        self.confidence_threshold = confidence_threshold
        self.distance_threshold = distance_threshold
        super().__init__(datapath, *args, **kwargs)

    def handle_mask(self, motion):
        center = motion[:, :-1, :2].mean(axis=1, keepdims=True)
        ball = motion[:, [-1], :2]
        mask = (motion[..., [2]] >= self.confidence_threshold).astype(np.float32)
        mask[:, [-1]] *= np.linalg.norm(center - ball, axis=-1, keepdims=True) < self.distance_threshold
        return mask

    def length_aug(self, motion, length, mask=None):
        MAX_LENGTH = 20 * 12
        MIN_LENGTH = 20 * 4
        while True and length > 1:
            start, end = np.random.randint(0, length + 1), np.random.randint(0, length + 1)
            if start > end:
                start, end = end, start
            if end - start > MIN_LENGTH and end - start < MAX_LENGTH:
                break
        motion = motion[start:end]
        mask = mask[start:end] if mask is not None else None
        new_length = end - start
        return motion, new_length, mask