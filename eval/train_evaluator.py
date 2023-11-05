import torch
from data_loaders.dataset_utils import get_dataset_loader_from_args
from eval.evaluator import create_evaluator
from train.train_platforms import get_train_platform
from train.trainer import Trainer
from utils.fixseed import fixseed
from utils.parser_utils import train_evaluator_args
from utils import dist_utils
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform, get_train_platform  # The platform classes are required when evaluating the train_platform argument

KL_DIV_LOSS_WEIGHT = 0.00001


def kl_div_loss(mu, logvar, **kwargs):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


mse = torch.nn.MSELoss()


class EvaluatorTrainer(Trainer):
    def calculate_loss(self, motion, cond):
        res = self.model(motion, cond)
        mu, logvar, recon_motion = res["mu"], res["sigma"], res["recon_motion"]
        loss = mse(recon_motion, motion)
        loss += KL_DIV_LOSS_WEIGHT * kl_div_loss(mu, logvar)
        return loss


def main():
    args = train_evaluator_args()
    fixseed(args.seed)
    dist_utils.setup_dist(args.device)

    print("Creating data loader...")
    data = get_dataset_loader_from_args(args, batch_size=args.train_batch_size)

    print("Creating model...")
    model = create_evaluator(args)
    model.to(dist_utils.dev())

    print("Training...")
    trainer = EvaluatorTrainer(args, model, data, get_train_platform(args))
    trainer.train()


if __name__ == "__main__":
    main()
