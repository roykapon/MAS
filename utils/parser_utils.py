from argparse import ArgumentParser
import argparse
from copy import deepcopy
import os
import json
from data_loaders.dataset_utils import get_dataset_config


def load_from_model(model_path):
    args_path = os.path.join(os.path.dirname(model_path), "args.json")
    assert os.path.exists(args_path), f"Arguments json file: {args_path} was not found!"
    with open(args_path, "r") as fr:
        return json.load(fr)


def parse_and_load_from_path(*path_names):
    parser = ArgumentParser(add_help=False)
    for path_name in path_names:
        parser.add_argument(f"--{path_name}", default=argparse.SUPPRESS, type=str)
    args, _ = parser.parse_known_args()
    for path_name in path_names:
        if hasattr(args, path_name):
            return load_from_model(getattr(args, path_name))
    return {}


def update_defaults(group, args):
    for action in group._group_actions:
        if action.dest in args:
            action.default = args[action.dest]


def parse_and_load_from_dataset():
    parser = ArgumentParser(add_help=False)
    group = parser.add_argument_group("dataset")
    group.add_argument("--dataset", default="nba", choices=["nba", "horse", "gymnastics"], type=str, help="Dataset name (choose from list).")
    update_defaults(group, parse_and_load_from_path("model_path", "resume_checkpoint", "evaluator_path"))
    dataset = parser.parse_known_args()[0].dataset
    return vars(get_dataset_config(dataset))


def add_base_options(parser: ArgumentParser):
    group = parser.add_argument_group("base")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=0, type=int, help="Seed for all random sampling.")


def add_diffusion_options(parser: ArgumentParser):
    group = parser.add_argument_group("diffusion")
    group.add_argument("--model_mean_type", default="x_start", type=str, choices=["x_start", "epsilon"], help="Defines the model's prediction type. Can be 'x_start' or 'epsilon'.")
    group.add_argument("--noise_schedule", default="cosine_tau_2", type=str, help="Noise schedule type. Can be 'cosine', 'linear' or 'cosine_tau_{n}' (cosine(normalize(t))^n).")
    group.add_argument("--diffusion_steps", default=100, type=int, help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Controls the variance type of the backwards posterior.")
    update_defaults(group, parse_and_load_from_path("model_path", "resume_checkpoint"))


def add_model_options(parser: ArgumentParser):
    group = parser.add_argument_group("model")
    group.add_argument("--arch", default="trans_enc", choices=["trans_enc", "trans_dec", "gru"], type=str, help="Architecture of the model.")
    group.add_argument("--emb_trans_dec", default=False, type=bool, help="For trans_dec architecture only, if true, will inject condition as a class token (in addition to cross-attention).")
    group.add_argument("--layers", default=8, type=int, help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int, help="Transformer/GRU width.")
    group.add_argument("--num_heads", default=4, type=int, help="Number of attention heads.")
    group.add_argument("--dropout", default=0.1, type=float, help="Dropout rate.")
    group.add_argument("--activation", default="gelu", type=str, help="Activation function.")
    group.add_argument("--ff_size", default=1024, type=int, help="Feed-forward size.")
    group.add_argument("--cond_mask_prob", default=0.1, type=float, help="The probability of masking the condition when applying classifier-free guidance learning.")
    update_defaults(group, parse_and_load_from_path("model_path", "resume_checkpoint"))


def add_evaluator_options(parser: ArgumentParser):
    group = parser.add_argument_group("evaluator")
    group.add_argument("--e_num_layers", default=6, type=int, help="Number of layers.")
    group.add_argument("--e_latent_dim", default=256, type=int, help="Transformer width.")
    group.add_argument("--e_num_heads", default=4, type=int, help="Number of attention heads.")
    group.add_argument("--e_dropout", default=0.1, type=float, help="Dropout rate.")
    group.add_argument("--e_activation", default="gelu", type=str, help="Activation function.")
    group.add_argument("--e_ff_size", default=1024, type=int, help="Feed-forward size.")
    update_defaults(group, parse_and_load_from_path("evaluator_path"))


def add_data_options(parser: ArgumentParser):
    group = parser.add_argument_group("dataset")
    group.add_argument("--dataset", default="nba", choices=["nba", "horse", "gymnastics"], type=str, help="Dataset name (choose from list).")
    update_defaults(group, parse_and_load_from_path("model_path", "resume_checkpoint", "evaluator_path"))
    group.add_argument("--datapath", default="", type=str, help="The directory that holds the motions directory. If empty, will use defaults according to the specified dataset.")
    group.add_argument("--data_size", default=None, type=int, help="If specified, will use a subset of the dataset with the given size.")
    group.add_argument("--data_split", default="train", choices=["train", "test"], type=str, help="Which data split to use.")
    group.add_argument("--data_augmentations", default=[], nargs="+", type=str, choices=["shift_aug", "scale_aug", "length_aug", ""], help="Types of data augmentations to apply.")
    group.add_argument("--cond", default="no_cond", type=str, choices=["no_cond", "text", "action"], help="Type of conditioning to use.")
    update_defaults(group, parse_and_load_from_dataset())


def add_training_options(parser: ArgumentParser):
    group = parser.add_argument_group("training")
    group.add_argument("--save_dir", required=True, type=str, help="Path to save checkpoints and results.")
    group.add_argument("--overwrite_model", action="store_true", help="If True, will enable to use an already existing output_dir, while deleting the original content.")
    group.add_argument("--train_platform_type", default="NoPlatform", choices=["NoPlatform", "ClearmlPlatform", "TensorboardPlatform"], type=str, help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--lr", default=1e-5, type=float, help="Learning rate.")
    group.add_argument("--eval_during_training", action="store_true", help="If True, will run evaluation during training.")
    group.add_argument("--save_interval", default=50_000, type=int, help="Number of steps between each checkpoint save.")
    group.add_argument("--num_steps", default=600_000, type=int, help="Training will stop after the specified number of steps.")
    group.add_argument("--resume_checkpoint", default="", type=str, help="If not empty, will start from the specified checkpoint (path to <model_name>.pth file).")
    group.add_argument("--train_batch_size", default=32, type=int, help="Batch size during training.")
    update_defaults(group, parse_and_load_from_path("resume_checkpoint"))


def get_use_data():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--use_data", action="store_true", help="If specified, will use data / condition from dataset. You can override the conditions by specifying motion_length or text_prompt.")
    return parser.parse_known_args()[0].use_data


def add_sampling_options(parser: ArgumentParser):
    group = parser.add_argument_group("sampling")
    group.add_argument("--model_path", default="", type=str, help="Path to <checkpoint_name>.pth file to be loaded.")
    group.add_argument("--output_dir", default="", type=str, help="Path to results dir (automatically created by the script). " "If empty, will create an output dir inside the model_path directory.")
    group.add_argument("--num_samples", default=10, type=int, help="Number of samples to generate.")
    group.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    group.add_argument("--guidance_param", default=2.5, type=float, help="Specifies the classifier-free guidance scale parameter.")
    group.add_argument("--overwrite", action="store_true", help="If True, will enable to use an already existing output_dir, while deleting the original content.")
    group.add_argument("--use_data", action="store_true", help="If specified, will use data / condition from dataset. You can override the conditions by specifying motion_length or text_prompt.")
    group.add_argument("--show_input_motions", action="store_true", help="If True, will show the motions sampled from the data.")
    group.add_argument("--motion_length", default=6.0 if not get_use_data() else None, type=float, help="The length of the sampled motion in seconds.")
    group.add_argument("--text_prompt", default=None, type=str, help="A text prompt to use for conditioning.")


def add_mas_options(parser: ArgumentParser):
    group = parser.add_argument_group("mas")
    group.add_argument("--num_views", default=7, type=int, help="Number of views optimize with.")


def add_ablation_options(parser: ArgumentParser):
    group = parser.add_argument_group("ablation")
    group.add_argument("--ablation_name", default="dreamfusion", type=str, choices=["mas", "dreamfusion", "dreamfusion_annealed", "no_3d_noise", "prominent_angle", "prominent_angle_no_3d_noise", "random_angles"], help="Which ablation to run.")


def add_evaluation_options(parser: ArgumentParser):
    group = parser.add_argument_group("evaluation")
    group.add_argument("--subjects", default="motionBert ElePose train_data model mas dreamfusion no_3d_noise", type=str, help="Subjects to evaluate, separated by spaces.")
    group.add_argument("--vis_subjects", default="", type=str, help="Subjects to visualize, separated by spaces. For example: 'motionBert mas'.")
    group.add_argument("--num_visualize_samples", default=4, type=int, help="Number of samples to visualize for each subject.")
    group.add_argument("--metrics_names", default="fid diversity precision recall", type=str, help="Metrics to measure, separated by spaces.")
    group.add_argument("--evaluator_path", default="save/evaluator/nba_v2/no_aug/checkpoint_1000000.pth", type=str, help="Path to <model_name>.pth file to use for the evaluator.")
    group.add_argument("--num_eval_iterations", default=10, type=int, help="Number of iterations to run evaluation loop.")
    group.add_argument("--angle_mode", default="uniform", type=str, choices=["uniform", "side", "hybrid"], help="Horizontal angle sampling mode for projecting the 3D motions.")
    group.add_argument("--eval_num_samples", default=1024, type=int, help="Number of samples to load for each subject during evaluation loop. If you wish to compare to the results presented in the paper, do not change the default value.")


def add_evaluator_sampling_options(parser: ArgumentParser):
    group = parser.add_argument_group("evaluator_sampling")
    group.add_argument("--evaluator_sampling_mode", default="reconstruct", type=str, choices=["reconstruct", "generate"], help="Sampling mode for the evaluator.")


def add_model_diffusion_data_options(parser):
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)


def get_eval_during_training():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--eval_during_training", action="store_true", help="If True, will run evaluation during training.")
    return parser.parse_known_args()[0].eval_during_training


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_model_diffusion_data_options(parser)
    add_training_options(parser)

    if get_eval_during_training():
        add_sampling_options(parser)
        add_mas_options(parser)
        add_evaluator_options(parser)
        add_evaluation_options(parser)
        args = parser.parse_args()
        evaluator_args = deepcopy(args)
        evaluator_args.model_path = args.evaluator_path
        evaluator_args.subjects, evaluator_args.vis_subjects, evaluator_args.metrics_names = evaluator_args.subjects.split(" "), evaluator_args.vis_subjects.split(" "), evaluator_args.metrics_names.split(" ")
        return args, evaluator_args

    return parser.parse_args(), None


def train_evaluator_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_evaluator_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def sample_evaluator_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_sampling_options(parser)
    add_evaluator_options(parser)
    add_data_options(parser)
    add_evaluator_sampling_options(parser)
    args = parser.parse_args()
    args.batch_size = args.num_samples
    args.data_size = args.num_samples
    return args


def evaluate_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_sampling_options(parser)
    add_mas_options(parser)
    add_model_diffusion_data_options(parser)
    add_evaluator_options(parser)
    add_evaluation_options(parser)
    args = parser.parse_args()
    evaluator_args = deepcopy(args)
    evaluator_args.model_path = args.evaluator_path
    evaluator_args.subjects, evaluator_args.vis_subjects, evaluator_args.metrics_names = evaluator_args.subjects.split(" "), evaluator_args.vis_subjects.split(" "), evaluator_args.metrics_names.split(" ")
    return args, evaluator_args


def generate_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_sampling_options(parser)
    add_model_diffusion_data_options(parser)
    args = parser.parse_args()
    args.batch_size = args.num_samples
    args.data_size = args.num_samples
    return args


def mas_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_sampling_options(parser)
    add_mas_options(parser)
    add_model_diffusion_data_options(parser)
    args = parser.parse_args()
    args.batch_size = args.num_samples
    args.data_size = args.num_samples
    return args


def ablation_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_sampling_options(parser)
    add_mas_options(parser)
    add_ablation_options(parser)
    add_model_diffusion_data_options(parser)
    args = parser.parse_args()
    args.batch_size = args.num_samples
    args.data_size = args.num_samples
    return args


def evaluation_parser():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_sampling_options(parser)
    add_mas_options(parser)
    add_evaluation_options(parser)
    return parser.parse_args()
