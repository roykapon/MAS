from copy import deepcopy
import os
import torch
from diffusion.gaussian_diffusion import GaussianDiffusion
from diffusion.resample import create_named_schedule_sampler
from eval.evaluate import evaluate
from train.trainer import Trainer
from utils.fixseed import fixseed
from utils.parser_utils import train_args
from utils import dist_utils
from data_loaders.dataset_utils import get_dataset_loader_from_args
from utils.model_utils import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform, get_train_platform  # The platform classes are required when evaluating the train_platform argument


def sum_batched(tensor: torch.Tensor, keepdim=False):
    return tensor.sum(dim=list(range(1, len(tensor.shape))), keepdim=keepdim)


def masked_weighted_l2(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor):
    # assuming a.shape == b.shape == mask.shape == bs, J, Jdim, seqlen
    loss = (a - b) ** 2
    loss = sum_batched(loss * mask) * weights
    unmasked_elements = torch.clamp(sum_batched(mask), min=1)
    return loss / unmasked_elements


class DiffusionTrainer(Trainer):
    def __init__(self, diffusion: GaussianDiffusion, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion = diffusion
        self.schedule_sampler_type = "uniform"
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)

    def calculate_loss(self, motion: torch.Tensor, cond):
        t, weights = self.schedule_sampler.sample(motion.shape[0], self.device)
        noise = torch.randn_like(motion)
        x_t = self.diffusion.q_sample(motion, t, noise)
        model_output = self.model(x=x_t, timesteps=t, **cond)
        loss = masked_weighted_l2(motion, model_output, cond["y"]["mask"], weights).mean()
        return loss


class DiffusionTrainerEval(DiffusionTrainer):
    def __init__(self, *args, evaluator_args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluator_args = evaluator_args

    def evaluate(self):
        if self.args.eval_during_training:
            model_path = os.path.join(self.args.save_dir, f"checkpoint_{self.step}.pth")
            current_args = deepcopy(self.args)
            setattr(current_args, "model_path", model_path)
            metrics, _ = evaluate(self.evaluator_args, current_args)
            for subject, subject_metrics in metrics.items():
                for metric_name, metric_value in subject_metrics.items():
                    self.train_platform.report_scalar(f"{subject}_{metric_name}", metric_value, self.step, group_name=metric_name)


def main():
    args, evaluator_args = train_args()
    fixseed(args.seed)
    dist_utils.setup_dist(args.device)

    print("Loading data...")
    data = get_dataset_loader_from_args(args, batch_size=args.train_batch_size)

    print("Creating model...")
    model, diffusion = create_model_and_diffusion(args)
    model.to(dist_utils.dev())

    print("Training...")
    trainer = DiffusionTrainerEval(diffusion, args, model, data, get_train_platform(args), evaluator_args = evaluator_args)
    trainer.train()


if __name__ == "__main__":
    main()
