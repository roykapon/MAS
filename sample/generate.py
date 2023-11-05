import torch
from tqdm import tqdm
from sample.sampler import DiffusionSampler
from utils.fixseed import fixseed
from utils.parser_utils import generate_args


class Generator(DiffusionSampler):
    def task_name(self):
        return "sample"

    @torch.no_grad()
    def __call__(self, save=True, visualize=True, model_kwargs=None, progress=True):
        self.load_model_kwargs(model_kwargs)
        x_t = self.sample_initial_xt()
        schedule = range(self.num_timesteps - 1, 0, -1)
        for t in tqdm(schedule) if progress else schedule:
            t_batch = torch.full((self.shape[0],), t, dtype=torch.long, device=self.device)
            sample = self.model(x_t, t_batch, **self.model_kwargs)
            x_t = self.sample_xt(sample, x_t, t)

        sample = self.inverse_transform(sample)
        if save or visualize:
            self.setup_dir()
        if save:
            self.save_motions(sample)
        if visualize:
            self.visualize(sample, "result")
        return sample


def main():
    args = generate_args()
    fixseed(args.seed)
    generator = Generator(args)
    generator()


if __name__ == "__main__":
    main()
