from data_loaders.dataset_utils import get_dims, get_num_actions
from model.mdm import MDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps


def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith("clip_model.") for k in missing_keys])


def create_model_and_diffusion(args):
    model = MDM(**get_model_args(args))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args(args):
    return {
        "njoints": get_dims(args.dataset)[0],
        "nfeats": get_dims(args.dataset)[1],
        "num_actions": get_num_actions(args.dataset),
        "latent_dim": args.latent_dim,
        "ff_size": args.ff_size,
        "num_layers": args.layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "activation": args.activation,
        "cond_mode": args.cond,
        "cond_mask_prob": args.cond_mask_prob,
        "arch": args.arch,
        "emb_trans_dec": args.emb_trans_dec,
        "clip_version": "ViT-B/32",
    }


def create_gaussian_diffusion(args):
    return SpacedDiffusion(
        use_timesteps=space_timesteps(args.diffusion_steps, [args.diffusion_steps]),
        betas=gd.get_named_beta_schedule(args.noise_schedule, args.diffusion_steps, 1.0),
        model_mean_type={"epsilon": gd.ModelMeanType.EPSILON, "x_start": gd.ModelMeanType.START_X, "previous_x": gd.ModelMeanType.PREVIOUS_X}[args.model_mean_type],
        model_var_type=(gd.ModelVarType.FIXED_LARGE if not args.sigma_small else gd.ModelVarType.FIXED_SMALL),
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
    )
