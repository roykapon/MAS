from types import SimpleNamespace
import numpy as np
from data_loaders.base_dataset import collate_with_mask
from data_loaders.gymnastics.dataset import GymnasticsDataset
from data_loaders.gymnastics.skeleton import LANDMARKS, SKELETON, TO_HUMANML_NAMES


GYMNASTICS_CONFIG = SimpleNamespace(
    **{
        "landmarks": LANDMARKS,
        "skeleton": SKELETON,
        "fps": 20,
        "dims": (18, 2),
        "datapath": "./dataset/gymnastics",
        "distance": 7,
        "sample_elevation_angle": lambda: np.pi / 16,
        "extract_trajectory": lambda motion: np.mean(motion[..., :-1, [0, 2]], axis=1),
        "visualization_scale": 0.66,
        "cond_mode": "no_cond",
        "confidence_threshold": 0.01,
        "distance_threshold": 3,
        "data_augmentations": ["length_aug"],
        "dataset_class": GymnasticsDataset,
        "collate_fn": collate_with_mask,
        "mean_path": "data_loaders/gymnastics/Mean.npy",
        "std_path": "data_loaders/gymnastics/Std.npy",
        "to_humanml": TO_HUMANML_NAMES,
    }
)
