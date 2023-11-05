import argparse
import os
from visualize import vis_utils
import shutil
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="stick figure mp4 file to be rendered.")
    parser.add_argument("--skeleton_type", type=str, default="nba", choices=["nba", "motionbert", "elepose", "gymnastics"])
    parser.add_argument("--cuda", type=bool, default=True, help="")
    parser.add_argument("--device", type=int, default=0, help="")
    parser.add_argument("--num_smplify_iters", type=int, default=150, help="")
    params = parser.parse_args()

    assert params.input_path.endswith(".mp4")
    sample_i = int(os.path.basename(params.input_path).split(".")[0].split("_")[-1])
    npy_path = os.path.join(os.path.dirname(params.input_path), "results.npy")
    assert os.path.exists(npy_path)

    npy2obj = vis_utils.npy2obj(npy_path, params.skeleton_type, sample_i, device=params.device, cuda=params.cuda, num_smplify_iters=params.num_smplify_iters)

    results_dir = params.input_path.replace(".mp4", "_obj")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)
    for frame_i in tqdm(range(npy2obj.length)):
        npy2obj.save_obj(os.path.join(results_dir, f"frame{frame_i:03d}.obj"), frame_i)
    print(f"Saved obj files to [{os.path.abspath(results_dir)}]")

    out_npy_path = params.input_path.replace(".mp4", "_smpl_params.npy")
    npy2obj.save_npy(out_npy_path)
    print(f"Saved SMPL params to [{os.path.abspath(out_npy_path)}]")


if __name__ == "__main__":
    main()
