import argparse
import os
import numpy as np
from data_loaders.dataset_utils import get_landmarks
from data_loaders.nba.skeleton import TO_HUMANML_NAMES as NBA_TO_HUMANML_NAMES
from data_loaders.gymnastics.skeleton import TO_HUMANML_NAMES as GYMNASTICS_TO_HUMANML_NAMES
from utils.math_utils import qbetween_np, qrot_np
from data_loaders.humanml.skeleton import LANDMARKS
from eval.eval_motionBert import convert_motionBert_skeleton


def rotate(motion):
    root_pos_init = motion[0]
    right_shoulder_index, left_shoulder_index = LANDMARKS.index("right_shoulder"), LANDMARKS.index("left_shoulder")
    across = root_pos_init[right_shoulder_index] - root_pos_init[left_shoulder_index]
    across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init**2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(motion.shape[:-1] + (4,)) * root_quat_init

    motion = qrot_np(root_quat_init, motion)
    motion[..., 2] = -motion[..., 2]
    return motion


def convert_skeleton_by_skeleton_type(motions, skeleton_type):
    landmarks = get_landmarks(skeleton_type)
    if skeleton_type == "gymnastics":
        motions_new_skeleton = convert_skeleton(motions, landmarks, GYMNASTICS_TO_HUMANML_NAMES)
        motions_new_skeleton = np.concatenate([motions_new_skeleton, motions[:, :, [-1]]], axis=2)  # add ball "joint"
        return motions_new_skeleton
    elif skeleton_type in ["nba", "motionbert", "elepose"]:
        return convert_skeleton(motions, landmarks, NBA_TO_HUMANML_NAMES)


def convert_skeleton(motions, landmarks, joint_to_humanml_names):
    new_motions = np.zeros((*motions.shape[:-2], len(LANDMARKS), 3))
    indices = np.array([(LANDMARKS.index(hml_name), landmarks.index(nba_name)) for hml_name, nba_name in joint_to_humanml_names])
    new_motions[:, :, indices[:, 0]] = motions[:, :, indices[:, 1]]

    covered_landmarks = [hml_landmark for hml_landmark, _ in joint_to_humanml_names]
    root_index, left_hip_index, right_hip_index = LANDMARKS.index("pelvis"), LANDMARKS.index("left_hip"), LANDMARKS.index("right_hip")
    if "pelvis" not in covered_landmarks:
        assert "left_hip" in covered_landmarks and "right_hip" in covered_landmarks, "pelvis could not be inferred from left_hip and right_hip"
        new_motions[:, :, root_index] = (new_motions[:, :, left_hip_index] + new_motions[:, :, right_hip_index]) / 2

    neck_index, left_shoulder_index, right_shoulder_index = LANDMARKS.index("neck"), LANDMARKS.index("left_shoulder"), LANDMARKS.index("right_shoulder")
    if "neck" not in covered_landmarks:
        assert "left_shoulder" in covered_landmarks and "right_shoulder" in covered_landmarks, "neck could not be inferred from left_shoulder and right_shoulder"
        new_motions[:, :, neck_index] = (new_motions[:, :, left_shoulder_index] + new_motions[:, :, right_shoulder_index]) / 2

    spine_1_index, spine_2_index, spine_3_index = LANDMARKS.index("spine1"), LANDMARKS.index("spine2"), LANDMARKS.index("spine3")
    if "spine1" not in covered_landmarks:
        new_motions[:, :, spine_1_index] = 0.75 * new_motions[:, :, root_index] + 0.25 * new_motions[:, :, neck_index]
    if "spine2" not in covered_landmarks:
        new_motions[:, :, spine_2_index] = 0.5 * new_motions[:, :, root_index] + 0.5 * new_motions[:, :, neck_index]
    if "spine3" not in covered_landmarks:
        new_motions[:, :, spine_3_index] = 0.25 * new_motions[:, :, root_index] + 0.75 * new_motions[:, :, neck_index]

    left_collar_index, right_collar_index = LANDMARKS.index("left_collar"), LANDMARKS.index("right_collar")
    if "left_collar" not in covered_landmarks:
        assert "left_shoulder" in covered_landmarks, "left_collar could not be inferred from left_shoulder"
        new_motions[:, :, left_collar_index] = 0.25 * new_motions[:, :, left_shoulder_index] + 0.75 * new_motions[:, :, neck_index]
    if "right_collar" not in covered_landmarks:
        assert "right_shoulder" in covered_landmarks, "right_collar could not be inferred from right_shoulder"
        new_motions[:, :, right_collar_index] = 0.25 * new_motions[:, :, right_shoulder_index] + 0.75 * new_motions[:, :, neck_index]

    return new_motions


def load_motions(motions_path, skeleton_type):
    data = np.load(motions_path, allow_pickle=True)
    if skeleton_type in ["nba", "gymnastics"]:
        motions = data.item()["motions"]
        return motions, data.item()["model_kwargs"]["y"]["lengths"]
    elif skeleton_type in ["motionbert"]:
        motions = data
        motions[..., 1] *= -1
        motions *= 2
        motions = convert_motionBert_skeleton(motions)
        motions = motions[:: 60 // 20]
        return motions[np.newaxis], np.array([motions.shape[0]])
    elif skeleton_type in ["elepose"]:
        motions = data
        motions = motions[:: 60 // 20]
        motions = convert_motionBert_skeleton(motions)
        return motions[np.newaxis], np.array([motions.shape[0]])


def convert(motions_path, skeleton_type):
    motions, lengths = load_motions(motions_path, skeleton_type)
    hml_motions = convert_skeleton_by_skeleton_type(motions, skeleton_type)
    hml_motions = np.stack([rotate(hml_motion) for hml_motion in hml_motions])
    return hml_motions, lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--motions_path", type=str)
    parser.add_argument("-s", "--skeleton_type", type=str, default="nba", choices=["nba", "motionbert", "elepose", "gymnastics"])
    parser.add_argument("-v", "--vis_dir", type=str, default="visualize/examples")
    parser.add_argument("--save", action="store_true")

    args = parser.parse_args()
    hml_motions, lengths = convert(args.motions_path, args.skeleton_type)
    hml_motions_reshaped = hml_motions.transpose(0, 2, 3, 1)  # (n_motions, n_frames, n_feats, 3) -> (n_motions, n_feats, 3, n_frames)
    print(hml_motions_reshaped.shape)

    hml_results = {"motion": hml_motions_reshaped, "lengths": lengths}

    save_path = os.path.join(args.vis_dir, args.motions_path.replace(".npy", "_hml"))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if args.save:
        np.save(save_path, hml_results)
        print(f"saved to {save_path}.npy")

    from utils.plot_script import plot_motion

    for i in range(hml_motions.shape[0]):
        plot_motion(f"{save_path}_{i}", hml_motions[i, : lengths[i]], dataset="humanml", rotate=True)


if __name__ == "__main__":
    main()