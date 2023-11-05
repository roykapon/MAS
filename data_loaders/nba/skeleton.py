LANDMARKS = [
    "Hip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
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

SKELETON_NAMES = [
    ["LWrist", "LElbow", "LShoulder", "Neck"],
    ["RWrist", "RElbow", "RShoulder", "Neck"],
    ["LAnkle", "LKnee", "LHip", "Hip"],
    ["RAnkle", "RKnee", "RHip", "Hip"],
    ["Hip", "Neck", "Nose", "Head"],
]

SKELETON = [[LANDMARKS.index(skeleton_name) for skeleton_name in sub_skeleton_names] for sub_skeleton_names in SKELETON_NAMES]

ROOT_INDEX = LANDMARKS.index("Hip")

MIRRORED_LANDMARKS = {
    "Nose": "Nose",  # 0
    "LEye": "REye",  # 1
    "REye": "LEye",  # 2
    "LEar": "REar",  # 3
    "REar": "LEar",  # 4
    "LShoulder": "RShoulder",  # 5
    "RShoulder": "LShoulder",  # 6
    "LElbow": "RElbow",  # 7
    "RElbow": "LElbow",  # 8
    "LWrist": "RWrist",  # 9
    "RWrist": "LWrist",  # 10
    "LHip": "RHip",  # 11
    "RHip": "LHip",  # 12
    "LKnee": "RKnee",  # 13
    "RKnee": "LKnee",  # 14
    "LAnkle": "RAnkle",  # 15
    "RAnkle": "LAnkle",  # 16
    "Head": "Head",  # 17
    "Neck": "Neck",  # 18
    "Hip": "Hip",  # 19
    "LBigToe": "RBigToe",  # 20
    "RBigToe": "LBigToe",  # 21
    "LSmallToe": "RSmallToe",  # 22
    "RSmallToe": "LSmallToe",  # 23
    "LHeel": "RHeel",  # 24
    "RHeel": "LHeel",  # 25
}

MIRRORED_INDICES = [LANDMARKS.index(MIRRORED_LANDMARKS[landmark]) for landmark in LANDMARKS]

TO_HUMANML_NAMES = [
    ("pelvis", "Hip"),
    ("right_hip", "RHip"),
    ("right_knee", "RKnee"),
    ("right_ankle", "RAnkle"),
    ("left_hip", "LHip"),
    ("left_knee", "LKnee"),
    ("left_ankle", "LAnkle"),
    ("neck", "Neck"),
    ("head", "Head"),
    ("left_shoulder", "LShoulder"),
    ("left_elbow", "LElbow"),
    ("left_wrist", "LWrist"),
    ("right_shoulder", "RShoulder"),
    ("right_elbow", "RElbow"),
    ("right_wrist", "RWrist"),
    ("left_foot", "LAnkle"),
    ("right_foot", "RAnkle"),
]
