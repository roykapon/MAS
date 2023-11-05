from typing import List, Tuple, Callable
import numpy as np
from data_loaders.base_dataset import BaseDataset


class DatasetConfig:
    landmarks: List[str]
    skeleton: List[List[int]]
    fps: int
    dims: Tuple[int, int]
    datapath: str
    distance: float
    sample_elevation_angle: Callable[[], float]
    extract_trajectory: Callable[[np.array], np.array]
    visualization_scale: float
    cond_mode: str
    confidence_threshold: float
    distance_threshold: float
    data_augmentations: List[str]
    dataset_class: BaseDataset
    collate_fn: Callable[[List, bool], Tuple]
    mean_path: str
    std_path: str
    to_humanml: List[Tuple[str, str]]
