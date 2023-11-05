from tqdm import tqdm
from data_loaders.dataset_utils import sample_distance, sample_vertical_angle, get_visualization_scale
from sample.generate import DiffusionSampler
from sample.sampler import get_title
from utils.fixseed import fixseed
from utils.math_utils import orthographic_projection, perspective_projection_z_detatched
import os
import numpy as np
import torch
from utils.parser_utils import mas_args
from utils.plot_script import plot_3d_motion
from torch.optim import Adam


OPTIMIZE_STEPS = 1000
OPTIMIZATION_THRESHOLD = 0.00000000005

mse = torch.nn.MSELoss()


def repeat_model_kwargs(model_kwargs, num_repeatitions):
    model_kwargs = {"y": {**model_kwargs["y"]}}
    model_kwargs["y"]["lengths"] = model_kwargs["y"]["lengths"].repeat(num_repeatitions)
    if "text" in model_kwargs["y"] and model_kwargs["y"]["text"] is not None:
        model_kwargs["y"]["text"] = model_kwargs["y"]["text"] * num_repeatitions
    if "scale" in model_kwargs["y"]:
        model_kwargs["y"]["scale"] = model_kwargs["y"]["scale"].repeat(num_repeatitions)
    return model_kwargs


class MAS(DiffusionSampler):
    def task_name(self):
        return "mas"

    def sample_angles(self):
        self.hor_angles = [-np.pi + i * 2 * np.pi / self.num_views for i in range(self.num_views)]
        self.ver_angles = [sample_vertical_angle(self.args.dataset) for _ in range(self.num_views)]

    def setup_motions_3d(self):
        self.num_views = self.args.num_views
        self.num_samples, self.n_joints = self.args.num_samples, self.model.njoints
        self.shape_3d = (self.num_samples, self.n_frames, self.n_joints, 3)
        self.shape = (self.args.num_samples * self.args.num_views, self.n_joints, self.n_feats, self.n_frames)
        self.motions_3d = torch.zeros(self.shape_3d, device=self.device, requires_grad=True)  # [bs, n_joints, 3, n_frames]
        self.distance = sample_distance(self.args.dataset)
        self.sample_angles()

    def sample_3d_noise(self, noise_3d=None):
        noise_3d = torch.randn_like(self.motions_3d) if noise_3d is None else noise_3d
        projected_noise = orthographic_projection(noise_3d, self.hor_angles, self.ver_angles)
        return projected_noise.permute(0, 1, 3, 4, 2)

    def handle_projected_motions(self, projected_motions: torch.Tensor):
        projected_motions = projected_motions.permute(0, 1, 3, 4, 2)
        return projected_motions * self.distance / get_visualization_scale(self.args.dataset)

    def project_motions_3d(self, motions_3d=None):
        motions_3d = self.motions_3d if motions_3d is None else motions_3d
        projected_motions = perspective_projection_z_detatched(motions_3d, self.hor_angles, self.ver_angles, self.distance)
        return self.handle_projected_motions(projected_motions)

    def setup_optimizer(self):
        self.optimizer = Adam([self.motions_3d], lr=0.01)
        self.schedule = range(self.num_timesteps - 1, 0, -1)

    def loss_fn(self, samples) -> torch.Tensor:
        return mse(self.project_motions_3d(), samples)

    def visualize_3d(self, motions_3d: torch.Tensor, title=""):
        motions_3d = motions_3d.detach().cpu().numpy()
        for i in range(self.num_samples):
            length = self.model_kwargs["y"]["lengths"][i].item()
            curr_motion_3d = motions_3d[i][:length]
            save_path = os.path.join(self.args.output_dir, f"{title}_{i}")
            plot_3d_motion(save_path, curr_motion_3d, dataset=self.args.dataset, title=get_title(self.model_kwargs, i), fps=self.fps, rotate=True, repeats=2)


    @torch.enable_grad()
    def optimize(self, samples):
        prev_loss = 10e10
        for i in range(OPTIMIZE_STEPS):
            loss_sample = self.loss_fn(samples)
            loss_sample.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if loss_sample >= prev_loss - OPTIMIZATION_THRESHOLD * prev_loss:
                break

            prev_loss = loss_sample

    def apply_model(self, x: torch.Tensor, t, **model_kwargs):
        # x: [num_views, batch_size, n_joints, n_feats, n_frames]
        timesteps = torch.full([self.num_samples * self.num_views], t, device=self.device, dtype=torch.long)
        x = x.reshape(self.shape)
        model_kwargs = repeat_model_kwargs(model_kwargs, self.num_views)
        pred = self.model(x, timesteps, **model_kwargs)
        if self.args.model_mean_type == "x_start":
            res = pred
        elif self.args.model_mean_type == "epsilon":
            res = self.diffusion._predict_xstart_from_eps(x, t, pred)
        res = res.reshape([self.num_views, self.num_samples, self.n_joints, self.n_feats, self.n_frames])
        return res

    def sample_xt(self, samples, x_t, t, eps=None):
        eps = self.sample_3d_noise() if eps is None else eps
        return self.coef1[t] * samples + self.coef2[t] * x_t + self.std[t] * eps

    def sample_initial_xt(self):
        return self.sample_3d_noise()

    def setup_step(self):
        pass

    @torch.no_grad()
    def __call__(self, model_kwargs=None, save=True, visualize=True, progress=True):
        if save or visualize:
            self.setup_dir()

        self.load_model_kwargs(model_kwargs)
        self.setup_motions_3d()
        self.setup_optimizer()

        self.x_t = self.sample_initial_xt()

        for self.t in (tqdm(self.schedule) if progress else self.schedule):
            self.setup_step()
            self.samples = self.inverse_transform(self.apply_model(self.x_t, self.t, **self.model_kwargs))

            self.optimize(self.samples)
            self.projected_motions = self.project_motions_3d()

            self.x_t = self.sample_xt(self.transform(self.projected_motions), self.x_t, self.t)

        if save:
            self.save_motions(self.motions_3d)
        if visualize:
            self.visualize_3d(self.motions_3d, title="result")

        return self.motions_3d


def main():
    args = mas_args()
    fixseed(args.seed)
    args.data_size = args.num_samples
    mas = MAS(args)
    mas()


if __name__ == "__main__":
    main()
