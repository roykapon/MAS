from copy import deepcopy
from math import ceil
import os
import random
import sys
import numpy as np
import torch
from tqdm import tqdm
from data_loaders.base_dataset import collate
from data_loaders.dataset_utils import get_visualization_scale, sample_vertical_angle, get_dataset_loader
from eval.metrics import calculate_frechet_distance, calculate_diversity, calculate_precision, calculate_recall
from sample.ablations import MAS_TYPES
from sample.generate import Generator
from sample.sampler import Sampler
from utils.fixseed import fixseed
from utils.math_utils import perspective_projection
from utils.parser_utils import evaluate_args
from eval.eval_motionBert import convert_motionBert_skeleton
from data_loaders.dataset_utils import sample_distance

ELEPOSE_SAMPLES_PATH = "dataset/nba/elepose_predictions"
MOTIONBERT_SAMPLES_PATH = "dataset/nba/motionBert_predictions"
DIVERSITY_TIMES = 200
PRECISION_AND_RECALL_K = 3


def convert_to_numpy(list_of_torch_tensors):
    return np.stack([tensor.cpu().numpy() for tensor in list_of_torch_tensors])


def project_to_random_angle(motion, dataset, mode):
    if mode == "uniform":
        hor_angle = np.random.uniform(-np.pi, np.pi)
    elif mode == "side":
        hor_angle = np.pi / 2
    elif mode == "hybrid":
        hor_angle = np.random.uniform(np.pi / 2 - np.pi / 4, np.pi / 2 + np.pi / 4)
    ver_angle = sample_vertical_angle(dataset)
    distance = sample_distance(dataset)
    projected_motion = perspective_projection(motion, [hor_angle], [ver_angle], distance)[0] * distance / get_visualization_scale(dataset)
    return projected_motion


from .test_evaluator import EvalSampler


class Evaluator:
    def __init__(self, args):
        self.encoder = EvalSampler(args)
        self.args = self.encoder.args

    def get_data_samples(self, split, **kwargs):
        return list(get_dataset_loader(self.args.dataset, split=split, data_size=self.args.eval_num_samples, batch_size=self.args.batch_size, **kwargs))

    def apply_on_data(self, func, split, **kwargs):
        return [func(input_motion, model_kwargs) for input_motion, model_kwargs in tqdm(self.get_data_samples(split, **kwargs))]

    def encode(self, motion, model_kwargs):
        return self.encoder.encode(motion, model_kwargs)[0]

    def encode_samples(self, iter):
        return [self.encode(motion, model_kwargs) for motion, model_kwargs in tqdm(iter)]

    @torch.no_grad()
    def get_model_samples(self, model_args, split="test"):
        # generate samples from the 2D diffusion model
        model_args = deepcopy(model_args)
        model_args.num_samples = model_args.batch_size
        model = Generator(model_args)

        def generate_motion(input_motion, model_kwargs):
            sample = model(save=False, visualize=False, model_kwargs=deepcopy(model_kwargs), progress=False)
            return model.transform(sample), model_kwargs

        return self.apply_on_data(generate_motion, split)

    @torch.no_grad()
    def get_mas_samples(self, mas_type_name, model_args, split="test"):
        model_args = deepcopy(model_args)
        model_args.num_samples = model_args.batch_size

        if mas_type_name in ["dreamfusion", "dreamfusion_annealed"]:
            model_args.num_views = 1

        model_3d: Sampler = MAS_TYPES[mas_type_name](model_args)

        def generate_motion(input_motion, model_kwargs):
            samples_3d = model_3d(save=False, visualize=False, model_kwargs=deepcopy(model_kwargs), progress=False)
            samples_2d = torch.stack([project_to_random_angle(sample_3d, self.args.dataset, self.args.angle_mode) for sample_3d in samples_3d]).permute(0, 2, 3, 1)
            return model_3d.transform(samples_2d), model_kwargs

        return self.apply_on_data(generate_motion, split)

    def get_3d_samples(self, samples_path, scale=1, flip=False, fps_ratio=1):
        motions = []
        file_names = os.listdir(samples_path)
        random.shuffle(file_names)  # Shuffling instead of random sampling can increase diversity and recall. Should not hurt FID and precision by much.
        for sample_file in tqdm(file_names[: self.args.eval_num_samples]):
            motion = np.load(os.path.join(samples_path, sample_file))
            motion = convert_motionBert_skeleton(motion)
            motion = motion[::fps_ratio]  # fix fps
            motion[..., 1] *= -1 if flip else 1  # flip y axis
            motion *= scale  # scale
            motions.append(motion)

        lengths = [len(motion) for motion in motions]
        masks = [None for motion in motions]
        motions_batched = [zip(motions[i : i + self.args.batch_size], lengths[i : i + self.args.batch_size], masks[i : i + self.args.batch_size]) for i in range(0, len(motions), self.args.batch_size)]
        motions_batched = [collate(batch) for batch in motions_batched]
        motions_batched = [(project_to_random_angle(motion_batch.permute(0, 3, 1, 2), self.args.dataset, self.args.angle_mode).permute(0, 2, 3, 1), model_kwargs) for motion_batch, model_kwargs in motions_batched]
        motions_batched = [(self.encoder.transform(motion_batch.to(self.encoder.device)), model_kwargs) for motion_batch, model_kwargs in motions_batched]

        return motions_batched

    def get_multiple_samples(self, subjects, model_args=None):
        return {subject: self.get_samples(subject, model_args) for subject in subjects}

    def get_samples(self, subject, model_args=None):
        if "test_data" == subject:
            return self.get_data_samples("test")

        elif "motionBert" == subject:
            return self.get_3d_samples(MOTIONBERT_SAMPLES_PATH, scale=1.7, flip=True, fps_ratio=2)  # get motionBert samples

        elif "ElePose" == subject:
            return self.get_3d_samples(ELEPOSE_SAMPLES_PATH, scale=1, flip=False, fps_ratio=2)  # get ElePose samples

        elif "train_data" == subject:
            return self.get_data_samples("train")

        elif "model" == subject:
            return self.get_model_samples(model_args)

        elif subject in MAS_TYPES.keys():
            return self.get_mas_samples(subject, model_args)

        else:
            raise ValueError(f"Unknown subject {subject}")

    def visualize_samples(self, motions):
        for vis_subject in self.args.vis_subjects:
            if vis_subject in motions:
                for motion, model_kwargs in motions[vis_subject][: ceil(self.args.num_visualize_samples / self.args.batch_size)]:
                    self.encoder.visualize(motion.to(self.encoder.device), f"{vis_subject}_sample", model_kwargs=model_kwargs, normalized=True)

    def evaluate_samples(self, samples, test_samples):
        encodings = convert_to_numpy(unbatch(self.encode_samples(samples)))
        test_encodings = convert_to_numpy(unbatch(self.encode_samples(test_samples)))

        all_metrics = {}
        if "fid" in self.args.metrics_names:
            all_metrics["fid"] = calculate_frechet_distance(encodings, test_encodings)

        if "diversity" in self.args.metrics_names:
            all_metrics["diversity"] = calculate_diversity(encodings, diversity_times=min(len(test_encodings) - 1, DIVERSITY_TIMES))

        if "precision" in self.args.metrics_names:
            all_metrics["precision"] = calculate_precision(encodings, test_encodings, k=PRECISION_AND_RECALL_K)

        if "recall" in self.args.metrics_names:
            all_metrics["recall"] = calculate_recall(encodings, test_encodings, k=PRECISION_AND_RECALL_K)

        return all_metrics

    def evaluate(self, model_args=None):
        all_samples = self.get_multiple_samples(self.args.subjects, model_args)
        self.visualize_samples(all_samples)

        test_samples = self.get_samples("test_data")
        all_metrics = {}

        for subject, samples in all_samples.items():
            metrics = self.evaluate_samples(samples, test_samples)
            all_metrics[subject] = metrics

        return all_metrics

    def evaluate_multiple_times(self, model_args=None, print=True, save=True):
        all_metrics = {subject: {metric_name: [] for metric_name in self.args.metrics_names} for subject in self.args.subjects}

        for i in range(self.args.num_eval_iterations):
            iter_metrics = self.evaluate(model_args)

            for subject, subject_iter_metrics in iter_metrics.items():
                for metric_name, metric in subject_iter_metrics.items():
                    all_metrics[subject][metric_name].append(metric)

        metrics_means, metrics_intervals = summarize_metrics(all_metrics, self.args)

        if print:
            print_table(metrics_means, metrics_intervals, self.args.subjects, self.args.metrics_names)

        if save:
            log_path = model_args.model_path.replace(".pth", f"_eval_{self.args.angle_mode}.txt")
            save_metrics(metrics_means, metrics_intervals, log_path, self.args.subjects, self.args.metrics_names)

        return metrics_means, metrics_intervals


def unbatch(batched_list):
    return sum([list(batch) for batch in batched_list], [])


def evaluate(evaluator_args, model_args):
    print(f"Evaluating [{model_args.model_path}]")
    evaluator = Evaluator(evaluator_args)
    return evaluator.evaluate_multiple_times(model_args)


def main():
    model_args, evaluator_args = evaluate_args()
    fixseed(evaluator_args.seed)
    evaluate(evaluator_args, model_args)


def summarize_metrics(all_metrics, evaluator_args):
    total_metrics = {subject: {metric_name: [] for metric_name in evaluator_args.metrics_names} for subject in evaluator_args.subjects}
    for subject, subject_all_metrics in all_metrics.items():
        for metric_name, subject_metric in subject_all_metrics.items():
            total_metrics[subject][metric_name].append(subject_metric)

    total_means = {subject: {metric_name: np.mean(metrics) for metric_name, metrics in subject_all_metrics.items()} for subject, subject_all_metrics in total_metrics.items()}
    total_stds = {subject: {metric_name: np.std(metrics) for metric_name, metrics in subject_all_metrics.items()} for subject, subject_all_metrics in total_metrics.items()}
    conf_intervals = {subject: {metric_name: 1.96 * std / np.sqrt(evaluator_args.num_eval_iterations) for metric_name, std in subject_stds.items()} for subject, subject_stds in total_stds.items()}

    return total_means, conf_intervals


def format_float(f):
    if f"{f:.2f}" == "0.00":
        return f"{f:.2e}"
    return f"{f:.2f}"


def print_table(total_means, conf_intervals, subjects, metrics_names):
    print(" " * 24 + "".join([f"{metric_name: <20}" for metric_name in (metrics_names)]))
    for subject in subjects:
        line = f"{subject: <24}"
        for metric_name in metrics_names:
            metrics = f"{format_float(total_means[subject][metric_name])}±{format_float(conf_intervals[subject][metric_name])}"
            line += f"{metrics: <20}"
        print(line)


def save_metrics(metrics_means, metrics_intervals, log_file_path, subjects, metrics_names):
    sys.stdout = open(log_file_path, "w")
    print_table(metrics_means, metrics_intervals, subjects, metrics_names)
    sys.stdout = sys.__stdout__
    print(f"Saved eval log to [{log_file_path}]")


if __name__ == "__main__":
    main()
