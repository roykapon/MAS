from typing import Any

import torch
from eval.evaluator import create_evaluator
from sample.sampler import Sampler, model_kwargs_to_device
from utils.parser_utils import sample_evaluator_args


class EvalSampler(Sampler):
    def task_name(self):
        return "eval_test"
    
    def load_model(self):
        self.VAE = create_evaluator(self.args)
        state_dict = torch.load(self.args.model_path, map_location=self.device)["model"]
        self.VAE.load_state_dict(state_dict)
        self.VAE.to(self.device)
        self.VAE.eval()

    @torch.no_grad()
    def reconstruct(self, visualize=True):
        self.load_model_kwargs()
        sample = self.VAE(self.input_motions, self.model_kwargs)["recon_motion"]

        if visualize:
            self.setup_dir()
            self.visualize(sample, "sample", normalized=True)
            self.visualize(self.input_motions, "input_motion", normalized=True)

    @torch.no_grad()
    def generate(self, visualize=True):
        self.load_model_kwargs()
        sample = self.VAE.decoder(torch.randn([self.args.num_samples, self.args.e_latent_dim], device=self.device), self.model_kwargs)
        if visualize:
            self.setup_dir()
            self.visualize(sample, "sample", normalized=True)

    @torch.no_grad()
    def encode(self, input_motions, model_kwargs):
        model_kwargs = model_kwargs_to_device(model_kwargs, self.device)
        input_motions = input_motions.to(self.device)
        return self.VAE.encoder(input_motions, model_kwargs)


def main():
    args = sample_evaluator_args()
    args.data_size = args.num_samples
    sampler = EvalSampler(args)
    if args.evaluator_sampling_mode == "reconstruct":
        sampler.reconstruct()
    elif args.evaluator_sampling_mode == "generate":
        sampler.generate()
    else:
        raise NotImplementedError(f"Invalid sampling mode: {args.evaluator_sampling_mode}")


if __name__ == "__main__":
    main()
