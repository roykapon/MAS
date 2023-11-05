from data_loaders.base_dataset import BaseDataset

OG_FPS = 60


class NBADataset(BaseDataset):
    def __init__(self, *args, fps, confidence_threshold, **kwargs):
        self.fps = fps
        self.confidence_threshold = confidence_threshold
        super().__init__(*args, **kwargs)

    def handle_mask(self, motion):
        return motion[..., [2]] >= self.confidence_threshold

    def preprocess_motion(self, motion):
        motion = super().preprocess_motion(motion)
        return motion[:: OG_FPS // self.fps]
