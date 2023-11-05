from types import SimpleNamespace
import numpy as np
from data_loaders.base_dataset import collate_with_text
from data_loaders.humanml.dataset import HunammlDataset
from data_loaders.humanml.skeleton import LANDMARKS, ROOT_INDEX, SKELETON


HUMANML_CONFIG = SimpleNamespace(
    **{
        "landmarks": LANDMARKS,
        "skeleton": SKELETON,
        "fps": 30,
        "dims": (len(LANDMARKS), 2),
        "datapath": "./dataset/projection",
        "distance": 3,
        "sample_elevation_angle": lambda: np.random.uniform(-np.pi / 16, np.pi / 8),
        "extract_trajectory": lambda motion: motion[:, ROOT_INDEX, [0, 2]],
        "visualization_scale": 0.5,
        "cond_mode": "text",
        "data_augmentations": ["length_aug"],
        "dataset_class": HunammlDataset,
        "collate_fn": collate_with_text,
        "mean_path": "data_loaders/humanml/Mean.npy",
        "std_path": "data_loaders/humanml/Std.npy",
    }
)
