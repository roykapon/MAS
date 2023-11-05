from abc import abstractmethod
import random
import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
from tqdm import tqdm

from data_loaders.tensors import collate_tensors, lengths_to_mask

TEST_SPLIT_SIZE = 0.1


class BaseDataset(data.Dataset):
    def __init__(self, datapath, mean_path, std_path, data_size=None, split="train", **kwargs):
        self.split = split

        self.data_dict = {}
        self.name_list = []

        motion_dir = pjoin(datapath, "motions")
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)

        files = os.listdir(motion_dir)

        split_index = int(len(files) * TEST_SPLIT_SIZE)
        files = files[:split_index] if split == "test" else files[split_index:]
        random.shuffle(files)

        for file_name in tqdm(files):
            motion = np.load(pjoin(motion_dir, file_name))
            name = file_name.split(".")[0]
            self.add_motion(motion, name)
            if len(self.data_dict) == data_size:
                break

    def add_motion(self, motion, name):
        self.name_list.append(name)
        motion = self.preprocess_motion(motion)
        self.data_dict[name] = {"motion": self.handle_motion(motion), "length": len(motion), "mask": self.handle_mask(motion)}

    def preprocess_motion(self, motion):
        return motion

    def handle_mask(self, motion):
        return motion[..., [2]]

    def handle_motion(self, motion):
        return self.transform(motion[..., :2])

    def transform(self, data):
        return (data - self.mean) / self.std

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def handle_item(self, motion, length, mask):
        return motion, length, mask

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        return self.handle_item(data["motion"], data["length"], data["mask"])


class AugDataset(BaseDataset):
    def __init__(self, *args, data_augmentations=[], **kwargs):
        self.data_augmentations = data_augmentations
        self.augmentations = []
        if "length_aug" in self.data_augmentations:
            self.setup_length_aug(**kwargs)
        if "scale_aug" in self.data_augmentations:
            self.setup_scale_aug(**kwargs)
        if "shift_aug" in self.data_augmentations:
            self.setup_shift_aug(**kwargs)
        super().__init__(*args, **kwargs)

    def setup_length_aug(self, **kwargs):
        self.augmentations.append(self.length_aug)

    def setup_scale_aug(self, scale_aug_variance, **kwargs):
        self.scale_aug_variance = scale_aug_variance
        self.augmentations.append(self.scale_aug)

    def setup_shift_aug(self, shift_aug_variance, **kwargs):
        self.shift_aug_variance = shift_aug_variance
        self.augmentations.append(self.shift_aug)

    def length_aug(self, motion, length, mask=None):
        while True and length > 1:
            start, end = np.random.randint(0, length + 1), np.random.randint(0, length + 1)
            if start > end:
                start, end = end, start
            if end - start > length // 2:
                break
        motion = motion[start:end]
        mask = mask[start:end] if mask is not None else None
        new_length = end - start
        return motion, new_length, mask

    def sample_scale(self):
        return np.random.normal(0, self.scale_aug_variance)

    def scale_aug(self, motion, length, mask=None):
        scale = self.sample_scale()
        motion = self.inv_transform(motion)
        motion[..., :2] *= np.e**scale
        motion = self.transform(motion)
        return motion, length, mask

    def shift_aug(self, motion, length, mask=None):
        shift = np.random.normal(0, self.shift_aug_variance, [2])
        motion = self.inv_transform(motion)
        motion[..., :2] += shift
        motion = self.transform(motion)
        return motion, length, mask

    def handle_item(self, motion, length, mask):
        for f in self.augmentations:
            motion, length, mask = f(motion, length, mask)
        return motion, length, mask


def collate(batch, uncond=True):
    motions, lengths, _ = zip(*batch)
    collate_motions = collate_tensors([torch.as_tensor(motion) for motion in motions])
    collate_lengths = torch.as_tensor(lengths)
    collate_masks = lengths_to_mask(collate_lengths, collate_motions.shape[1]).unsqueeze(-1).unsqueeze(-1).expand(collate_motions.shape)

    collate_motions = collate_motions.permute(0, 2, 3, 1)  # [bs, seq_len, n_joints, n_feats] -> [bs, n_joints, n_feats, seq_len]
    collate_masks = collate_masks.permute(0, 2, 3, 1)  # [bs, seq_len, n_joints] -> [bs, n_joints, 1, seq_len]
    return collate_motions, {"y": {"lengths": collate_lengths, "mask": collate_masks, "uncond": uncond}}


def collate_with_mask(batch, uncond=True):
    _, _, masks = zip(*batch)
    motion, model_kwargs = collate(batch, uncond=uncond)
    # The original mask is based on the confidence level of the pose estimator.
    # The new mask is a combination of the original mask and the mask based on the length of the sequence
    masks = collate_tensors([torch.as_tensor(mask) for mask in masks]).permute(0, 2, 3, 1)  # [bs, seq_len, n_joints] -> [bs, n_joints, 1, seq_len]
    model_kwargs["y"]["mask"] = model_kwargs["y"]["mask"] * masks
    return motion, model_kwargs


def collate_with_text(batch, uncond=True):
    motions, lengths, masks, texts = zip(*batch)
    collate_motions, model_kwargs = collate(zip(motions, lengths, masks), uncond=uncond)
    model_kwargs["y"]["text"] = texts
    return collate_motions, model_kwargs
