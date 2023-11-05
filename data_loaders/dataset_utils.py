from copy import deepcopy
from functools import partial
from os.path import join as pjoin
import numpy as np
from data_loaders.base_config import DatasetConfig
from data_loaders.gymnastics.config import GYMNASTICS_CONFIG
from data_loaders.horse.config import HORSE_CONFIG
from data_loaders.nba.config import NBA_CONFIG
from data_loaders.humanml.config import HUMANML_CONFIG
from torch.utils.data import DataLoader

CONFIGS = {
    "nba": NBA_CONFIG,
    "horse": HORSE_CONFIG,
    "gymnastics": GYMNASTICS_CONFIG,
    "humanml": HUMANML_CONFIG,
}


def get_dataset_config(dataset) -> DatasetConfig:
    if dataset not in CONFIGS:
        raise ValueError(f"Unsupported dataset [{dataset}]")
    else:
        return CONFIGS[dataset]


def get_skeleton(dataset):
    return get_dataset_config(dataset).skeleton


def get_landmarks(dataset):
    return get_dataset_config(dataset).landmarks


def get_fps(dataset):
    return get_dataset_config(dataset).fps


def get_dims(dataset):
    return get_dataset_config(dataset).dims


def sample_distance(dataset):
    return get_dataset_config(dataset).distance


def sample_vertical_angle(dataset):
    return get_dataset_config(dataset).sample_elevation_angle()


def get_trajectory(dataset, motion):
    return get_dataset_config(dataset).extract_trajectory(motion)


def get_visualization_scale(dataset):
    return get_dataset_config(dataset).visualization_scale


def get_cond_mode(dataset):
    return get_dataset_config(dataset).cond_mode


def get_dataset(dataset, **kwargs):
    return get_dataset_config(dataset).dataset_class(**kwargs)


def get_data_augmentations(dataset):
    return get_dataset_config(dataset).data_augmentations


def get_dataset_from_args(args, **kwargs):
    default_kwargs = vars(args)
    default_kwargs.update(**kwargs)
    return get_dataset(args.dataset, **default_kwargs)


def get_collate_fn(dataset):
    return get_dataset_config(dataset).collate_fn


def get_dataset_loader(dataset, batch_size, num_workers=0, **kwargs):
    default_kwargs = deepcopy(vars(get_dataset_config(dataset)))
    default_kwargs.update(**kwargs)
    dataset_object = get_dataset(dataset, **default_kwargs)
    collate_fn = partial(get_collate_fn(dataset), uncond=(default_kwargs["cond_mode"] == "no_cond"))
    return DataLoader(dataset_object, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=collate_fn)


def get_dataset_loader_from_args(args, **kwargs):
    default_args = deepcopy(vars(args))
    default_args.update(**kwargs)
    return get_dataset_loader(**default_args)


def get_datapath(dataset):
    return get_dataset_config(dataset).datapath


def get_mean(dataset):
    return np.load(get_dataset_config(dataset).mean_path)


def get_std(dataset):
    return np.load(get_dataset_config(dataset).std_path)


def get_num_actions(dataset):
    return getattr(get_dataset_config(dataset), "num_actions", 1)
