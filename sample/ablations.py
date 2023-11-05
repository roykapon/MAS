import numpy as np
import torch
from data_loaders.dataset_utils import sample_vertical_angle
from sample.mas import MAS, mse
from sample.dreamfusion import Dreamfusion, DreamfusionAnnealed
from utils.fixseed import fixseed
from utils.parser_utils import ablation_args

LIFTED_MOTION_WEIGHT = 1000


def weighted_loss(x, y, weight):
    return torch.mean(weight * (x - y) ** 2)


class MASNo3DNoise(MAS):
    def task_name(self):
        return "mas_no_3d_noise"

    def sample_3d_noise(self):
        return torch.randn_like(super().sample_3d_noise())


class MASProminentAngle(MAS):
    def sample_weights(self):
        self.weights = torch.ones([self.num_views], device=self.device, dtype=torch.float32)
        rand_index = torch.randint(0, self.num_views, [1], device=self.device)
        self.weights[rand_index] = LIFTED_MOTION_WEIGHT
        self.weights /= self.weights.sum()

    def task_name(self):
        return "mas_prominent_angle"

    def setup_optimizer(self):
        super().setup_optimizer()
        self.sample_weights()

    def loss_fn(self, samples):
        return weighted_loss(self.project_motions_3d(), samples, self.weights.view(-1, 1, 1, 1, 1))


class MASProminentAngleNo3DNoise(MASNo3DNoise, MASProminentAngle):
    def task_name(self):
        return "mas_prominent_angle_no_3d_noise"


class MASRandomAngles(MAS):
    def task_name(self):
        return "random_angles"

    def __init__(self, args):
        super().__init__(args)
        self.motions_3d_timesteps = []
        self.noises_timesteps = []

    def sample_angles(self):
        offset = np.random.rand() * 2 * np.pi
        self.hor_angles = [offset + (-np.pi) + i * 2 * np.pi / self.num_views for i in range(self.num_views)]
        self.ver_angles = [sample_vertical_angle(self.args.dataset) for _ in range(self.num_views)]

    def optimize(self, samples):
        super().optimize(samples)
        self.motions_3d_timesteps.append(self.motions_3d.clone().detach())

    def sample_initial_xt(self):
        self.motions_3d_timesteps = []
        self.noises_timesteps = []

        z = torch.randn_like(self.motions_3d)
        self.noises_timesteps.append(z)
        return self.sample_3d_noise(z)

    def sample_xt(self, samples, og_x_t, t):
        self.noises_timesteps.append(torch.randn_like(self.motions_3d))
        self.sample_angles()
        x_t = self.sample_3d_noise(self.noises_timesteps[0])
        for t_index, t_hat in enumerate(range(self.num_timesteps - 1, t - 1, -1)):
            x_t = super().sample_xt(self.transform(self.project_motions_3d(self.motions_3d_timesteps[t_index])), x_t, t_hat, self.sample_3d_noise(self.noises_timesteps[t_index + 1]))
        return x_t

    def sample_3d_noise(self, noise_3d=None):
        noise_3d = torch.randn_like(super().sample_3d_noise()) if noise_3d is None else noise_3d
        return super().sample_3d_noise(noise_3d)


MAS_TYPES = {
    "mas": MAS,
    "dreamfusion": Dreamfusion,
    "dreamfusion_annealed": DreamfusionAnnealed,
    "no_3d_noise": MASNo3DNoise,
    "prominent_angle": MASProminentAngle,
    "prominent_angle_no_3d_noise": MASProminentAngleNo3DNoise,
    "random_angles": MASRandomAngles,
}


def main():
    args = ablation_args()
    fixseed(args.seed)
    args.data_size = args.num_samples
    model_3d = MAS_TYPES[args.ablation_name](args)
    model_3d()


if __name__ == "__main__":
    main()
