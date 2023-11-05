from types import SimpleNamespace
import numpy as np
from data_loaders.base_dataset import collate_with_mask
from data_loaders.nba.dataset import NBADataset
from data_loaders.nba.skeleton import LANDMARKS, ROOT_INDEX, SKELETON, TO_HUMANML_NAMES


NBA_CONFIG = SimpleNamespace(
    **{
        "landmarks": LANDMARKS,
        "skeleton": SKELETON,
        "fps": 30,
        "dims": (16, 2),
        "datapath": "./dataset/nba",
        "distance": 7,
        "sample_elevation_angle": lambda: np.pi / 16,
        "extract_trajectory": lambda motion: motion[:, ROOT_INDEX, [0, 2]],
        "visualization_scale": 0.75,
        "cond_mode": "no_cond",
        "confidence_threshold": 0.3,
        "data_augmentations": [],
        "dataset_class": NBADataset,
        "collate_fn": collate_with_mask,
        "mean_path": "data_loaders/nba/Mean.npy",
        "std_path": "data_loaders/nba/Std.npy",
        "to_humanml": TO_HUMANML_NAMES,
    }
)
