from types import SimpleNamespace
import numpy as np
from data_loaders.base_dataset import collate_with_mask
from data_loaders.horse.dataset import HorseDataset
from data_loaders.horse.skeleton import HORSE_LANDMARKS, HORSE_SKELETON


HORSE_CONFIG = SimpleNamespace(
    **{
        "landmarks": HORSE_LANDMARKS,
        "skeleton": HORSE_SKELETON,
        "fps": 30,
        "dims": (17, 2),
        "datapath": "./dataset/horse",
        "distance": 7,
        "sample_elevation_angle": lambda: np.pi / 16,
        "extract_trajectory": lambda motion: np.mean(motion[..., [0, 2]], axis=1),
        "visualization_scale": 0.5,
        "cond_mode": "no_cond",
        "confidence_threshold": 0.5,
        "data_augmentations": ["length_aug"],
        "dataset_class": HorseDataset,
        "collate_fn": collate_with_mask,
        "mean_path": "data_loaders/horse/Mean.npy",
        "std_path": "data_loaders/horse/Std.npy",
    }
)
