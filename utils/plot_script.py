import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
from data_loaders.dataset_utils import get_skeleton, get_trajectory, get_visualization_scale


def plot_3d_motion(save_path, joints, dataset, title="", figsize=(5, 5), fps=20, radius=3, elev=30, azim=None, rotate=False, repeats=1, linewidth=1.0):
    matplotlib.use("Agg")
    title = "\n".join(wrap(title, 20))
    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)

    def init():
        ax.clear()
        ratio = figsize[0] / figsize[1]
        ax.set_xlim3d([-radius * ratio / 2, radius * ratio / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius * ratio / 3.0, radius * ratio / 3.0])
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz], [maxx, miny, minz]]
        verts = np.array(verts)
        verts[:, [0, 1, 2]] = verts[:, [0, 2, 1]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    # data: [seq_len, joints_num, 3]
    data = joints.copy().reshape(len(joints), -1, 3)
    data = np.concatenate([data] * repeats, axis=0)

    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"] * 10

    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset

    trajec = get_trajectory(dataset, motion=data)
    data[..., [0, 2]] -= trajec.reshape(-1, 1, 2)  # place subject in the center

    data[..., [0, 1, 2]] = data[..., [0, 2, 1]]  # switch y, z axes for matplotlib

    skeleton = get_skeleton(dataset)

    def update(index):
        init()
        ax.view_init(elev=elev, azim=azim if azim is not None else (index / 300 * 360 if rotate else 0))
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])
        ax.scatter(data[index, :, 0], data[index, :, 1], data[index, :, 2], s=3, c=colors_blue[: data.shape[1]])

        for i, (chain, color) in enumerate(zip(skeleton, colors_blue)):
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)
    save_animation(ani, save_path, fps)

    plt.close()


def plot_motion(save_path, joints, dataset, title="", mask=None, figsize=(5, 5), fps=20, **kwargs):
    skeleton = get_skeleton(dataset)
    data = joints.copy()

    data *= get_visualization_scale(dataset)

    joint_pairs = sum([list(zip(chain[1:], chain[:-1])) for chain in skeleton], [])
    lines = data[:, joint_pairs, :]  # [nframes, nlines, 2, 2]
    fig, ax = plt.subplots()

    wraped_title = "\n".join(wrap(title, 30))
    fig.suptitle(wraped_title, fontsize=10)

    colors = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A", "#80B79A"] * 10

    def init():
        ax.clear()
        ax.set_xlim(-figsize[0] / 2, figsize[0] / 2)
        ax.set_ylim(-figsize[1] / 2, figsize[1] / 2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_prop_cycle(color=colors)

    def animate(frame_i):
        init()
        frame_joint_colors = [color if joint_mask != 0 else "black" for color, joint_mask in zip(colors, mask[frame_i, :, 0])] if mask is not None else colors
        ax.scatter(data[frame_i, :, 0], data[frame_i, :, 1], s=10, c=frame_joint_colors[: data.shape[1]])
        for i in range(len(joint_pairs)):
            line_mask = min(mask[frame_i, joint_pairs[i][0], 0], mask[frame_i, joint_pairs[i][1], 0]) if mask is not None else 1
            frame_line_color = colors[i] if line_mask != 0 else "black"
            ax.plot(lines[frame_i, i, :, 0], lines[frame_i, i, :, 1], lw=1, c=frame_line_color)

    anim = FuncAnimation(fig, animate, interval=1000, frames=len(data))
    save_animation(anim, save_path, fps)

    plt.close()


def save_animation(anim: FuncAnimation, save_path, fps):
    save_path = f"{save_path}.mp4"
    print(f"saving to [{os.path.abspath(save_path)}]...", end="", flush=True)
    anim.save(save_path, fps=fps)
    print("\033[0K\r\033[0K\r", end="")
    print(f"saved to [{os.path.abspath(save_path)}]")
