import os
import random
import re
import numpy as np
from data_loaders.base_dataset import BaseDataset


DEFAULT_DATAPATH = "./dataset/projection"
MIN_MOTION_LENGTH = 10


class HunammlDataset(BaseDataset):
    def __init__(self, datapath=DEFAULT_DATAPATH, *args, **kwargs):
        super().__init__(datapath, *args, **kwargs)
        texsts_dir = os.path.join(datapath, "texts")
        for name in self.data_dict.keys():
            texts_and_tokens = open(os.path.join(texsts_dir, f"{name}.txt")).readlines()  # the text files are written in the format: text1. token1 \n text2. token2 \n...
            texts_and_tokens = [re.split("\.|#", text_and_tokens) for text_and_tokens in texts_and_tokens]
            self.data_dict[name]["texts"] = texts_and_tokens

    def add_motion(self, motion, name):
        if len(motion) > MIN_MOTION_LENGTH:
            super().add_motion(motion, name)
        else:
            print(f"Motion {name} is too short, skipping")

    def handle_mask(self, motion):
        return None

    def __getitem__(self, item):
        motion, length, mask = super().__getitem__(item)
        text_and_tokens = self.data_dict[self.name_list[item]]["texts"]
        text = random.choice(text_and_tokens)[0]
        return motion, length, mask, text