from data_loaders.nba.skeleton import LANDMARKS


CONVERT_SKELETON_NAMES = [
    "Hip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "Torse",
    "Neck",
    "Nose",
    "Head",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RShoulder",
    "RElbow",
    "RWrist",
]


CONVERT_NBA_V2_SKELETON = [CONVERT_SKELETON_NAMES.index(landmark) for landmark in LANDMARKS]


def convert_motionBert_skeleton(motion):
    return motion[:, CONVERT_NBA_V2_SKELETON]
