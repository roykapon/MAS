import numpy as np
from data_loaders.base_dataset import AugDataset


CONFIDENCE_THRESHOLD = 0.5
MAX_LENGTH = 30 * 20
MIN_LENGTH = 30 * 3


class HorseDataset(AugDataset):
    def add_motion(self, motion, name):
        if motion.shape[0] >= MIN_LENGTH:
            super().add_motion(motion, name)

    def handle_mask(self, motion):
        return motion[..., [2]] >= CONFIDENCE_THRESHOLD

    def length_aug(self, motion, length, mask=None):
        while True and length > 1:
            start, end = np.random.randint(0, length + 1), np.random.randint(0, length + 1)
            if start > end:
                start, end = end, start
            if end - start >= MIN_LENGTH and end - start < MAX_LENGTH:
                break
        motion = motion[start:end]
        mask = mask[start:end] if mask is not None else None
        new_length = end - start
        return motion, new_length, mask
