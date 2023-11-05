LANDMARKS = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

SKELETON_NAMES = [
    ["pelvis", "right_hip", "right_knee", "right_ankle", "right_foot"],
    ["pelvis", "left_hip", "left_knee", "left_ankle", "left_foot"],
    ["pelvis", "spine1", "spine2", "spine3", "neck", "head"],
    ["spine3", "right_collar", "right_shoulder", "right_elbow", "right_wrist"],
    ["spine3", "left_collar", "left_shoulder", "left_elbow", "left_wrist"],
]

SKELETON = [[LANDMARKS.index(name) for name in names] for names in SKELETON_NAMES]

ROOT_INDEX = LANDMARKS.index("pelvis")
