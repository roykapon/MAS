import numpy as np
import torch
from data_loaders.dataset_utils import sample_vertical_angle
from sample.mas import MAS
from utils.fixseed import fixseed
from utils.parser_utils import mas_args

NUM_ITERATIONS = 1000


class Dreamfusion(MAS):
    def task_name(self):
        return "dreamfusion"

    def sample_initial_xt(self):
        return torch.randn(self.shape, device=self.device)

    def sample_xt(self, samples, x_t, t):
        t = torch.full([self.num_views], t, device=self.device)
        return self.diffusion.q_sample(samples, t)

    @torch.enable_grad()
    def optimize(self, samples):
        loss_sample = self.loss_fn(samples)
        loss_sample.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def setup_optimizer(self):
        super().setup_optimizer()
        self.schedule = [np.random.randint(1, self.num_timesteps) for _ in range(NUM_ITERATIONS)]
        self.optimizer = torch.optim.Adam([self.motions_3d], lr=0.01)

    def setup_step(self):
        self.hor_angles = [np.random.uniform() for _ in range(self.num_views)]
        self.ver_angles = [sample_vertical_angle(self.args.dataset) for _ in range(self.num_views)]
        self.x_t = self.sample_xt(self.transform(self.project_motions_3d()), self.x_t, self.t)


class DreamfusionAnnealed(Dreamfusion):
    def task_name(self):
        return "dreamfusion_annealed"

    def setup_optimizer(self):
        super().setup_optimizer()
        self.schedule = [np.random.randint(self.num_timesteps - 1 - i * (self.num_timesteps / NUM_ITERATIONS), self.num_timesteps) for i in range(NUM_ITERATIONS)]


def main():
    args = mas_args()
    fixseed(args.seed)
    args.data_size = args.num_samples
    model_3d = Dreamfusion(args)
    model_3d()


if __name__ == "__main__":
    main()
