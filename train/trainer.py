import itertools
import json
import os
import torch
from tqdm import tqdm
from sample.sampler import model_kwargs_to_device
from train.train_platforms import NoPlatform, TrainPlatform
from utils import dist_utils
from torch.optim import AdamW
from torch.nn import Module


class Trainer:
    def __init__(self, args, model: Module, data, train_platform: TrainPlatform = NoPlatform(None)):
        self.args = args
        self.model = model
        self.data = data
        self.train_platform = train_platform
        self.optimizer = AdamW(model.parameters(), lr=args.lr)
        self.step = 0
        self.setup_device()
        self.setup_dir()
        self.loss = None
        self.mean_loss = None

        if self.args.resume_checkpoint:
            self.load_checkpoint()

    def load_checkpoint(self):
        checkpoint = torch.load(self.args.resume_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step = checkpoint["step"] + 1
        print(f"Loaded checkpoint: [{self.args.resume_checkpoint}]")

    def setup_device(self):
        dist_utils.setup_dist(self.args.device)
        self.device = dist_utils.dev()

    def setup_dir(self):
        if os.path.exists(self.args.save_dir) and not self.args.overwrite_model and (os.path.dirname(self.args.resume_checkpoint) != self.args.save_dir):
            raise FileExistsError(f"Save dir [{self.args.save_dir}] already exists.")
        elif not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        self.args_path = os.path.join(self.args.save_dir, "args.json")
        with open(self.args_path, "w") as fw:
            json.dump(vars(self.args), fw, indent=4, sort_keys=True)

    def calculate_loss(self, motion, cond) -> torch.Tensor:
        pass  # To be overridden

    def evaluate(self):
        pass  # To be overridden

    def train(self):
        for epoch in itertools.count(start=1):
            iter = tqdm(self.data)
            for motion, cond in iter:
                motion = motion.to(self.device)
                cond = model_kwargs_to_device(cond, self.device)
                self.optimizer.zero_grad()
                self.loss = self.calculate_loss(motion, cond)
                self.mean_loss = self.mean_loss * self.step / (self.step + 1) + self.loss.item() / (self.step + 1) if self.mean_loss else self.loss.item()

                self.loss.backward()
                self.optimizer.step()

                iter.set_description(f"epoch: {epoch:<3} | step: {self.step:<6} | mean_loss: {self.mean_loss:.4f}")
                self.train_platform.report_scalar(name="mean_loss", value=self.mean_loss, iteration=self.step, group_name="Loss")

                if self.step % self.args.save_interval == 0 and self.step != 0:
                    self.save_checkpoint()
                    if self.args.eval_during_training:
                        self.evaluate()

                self.step += 1
                if self.step > self.args.num_steps:
                    return

    def save_checkpoint(self):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
        }
        checkpoint_path = os.path.join(self.args.save_dir, f"checkpoint_{self.step}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to [{checkpoint_path}]")
