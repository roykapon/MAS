from visualize.convert_skeleton import convert
from model.rotation2xyz import Rotation2xyz
import numpy as np
from trimesh import Trimesh
import os
import torch
from visualize.simplify_loc2rot import joints2smpl


class npy2obj:
    def __init__(self, npy_path, skeleton_type, sample_idx, device=0, cuda=True, num_smplify_iters=150):
        self.npy_path = npy_path
        self.motions, self.lengths = convert(npy_path, skeleton_type)
        self.motion, self.length = self.motions[sample_idx], self.lengths[sample_idx]
        self.rot2xyz = Rotation2xyz(device="cpu" if not cuda else device)
        self.faces = self.rot2xyz.smpl_model.faces
        self.bs, self.nframes, self.njoints, self.nfeats = self.motions.shape
        self.opt_cache = {}
        self.sample_idx = sample_idx
        self.total_num_samples = self.motions.shape[0]

        simplify = joints2smpl(num_frames=self.length, device_id=device, cuda=cuda, num_smplify_iters=num_smplify_iters)
        self.rot_motions, _ = simplify.joint2smpl(self.motion[: self.length])

        self.vertices = self.rot2xyz(
            torch.tensor(self.rot_motions),
            mask=None,
            pose_rep="rot6d",
            translation=True,
            glob=True,
            jointstype="vertices",
            vertstrans=True,
        )

    def save_obj(self, save_path, frame_i):
        mesh = Trimesh(vertices=self.vertices[...,frame_i].squeeze().tolist(), faces=self.faces)
        with open(save_path, "w") as fw:
            mesh.export(fw, "obj")
        return save_path

    def save_npy(self, save_path):
        data_dict = {
            "motion": self.motion,
            "thetas": self.rot_motions[:-1],
            "faces": self.faces,
            "vertices": self.vertices,
            "length": self.length,
        }
        np.save(save_path, data_dict)
