
COCO_LANDMARKS = [
    "Nose", # 0
    "LeftEye", # 1
    "RightEye", # 2
    "LeftEar", # 3
    "RightEar", # 4
    "LeftShoulder", # 5
    "RightShoulder", # 6
    "LeftElbow", # 7
    "RightElbow", # 8
    "LeftWrist", # 9
    "RightWrist", # 10
    "LeftHip", # 11
    "RightHip", # 12
    "LeftKnee", # 13
    "RightKnee", # 14
    "LeftAnkle", # 15
    "RightAnkle" # 16
]

LANDMARKS = COCO_LANDMARKS + ["Ball"]

SKELETON_NAMES = [
    ["LeftShoulder", "LeftEar", "LeftEye", "Nose", "RightEye", "RightEar", "RightShoulder"],
    ["LeftShoulder", "LeftElbow", "LeftWrist"],
    ["RightShoulder", "RightElbow", "RightWrist"],
    ["LeftShoulder", "RightShoulder", "RightHip", "LeftHip", "LeftShoulder"],
    ["LeftHip", "LeftKnee", "LeftAnkle"],
    ["RightHip", "RightKnee", "RightAnkle"]
]

SKELETON = [[LANDMARKS.index(name) for name in names] for names in SKELETON_NAMES]

TO_HUMANML_NAMES = [
    # ("pelvis", ""),
    ("left_hip", "LeftHip"),
    ("right_hip", "RightHip"),
    # ("spine1", ""),
    ("left_knee", "LeftKnee"),
    ("right_knee", "RightKnee"),
    # ("spine2", ""),
    ("left_ankle", "LeftAnkle"),
    ("right_ankle", "RightAnkle"),
    # ("spine3", ""),
    ("left_foot", "LeftAnkle"),
    ("right_foot", "RightAnkle"),
    # ("neck", ""),
    # ("left_collar", ""),
    # ("right_collar", ""),
    ("head", "Nose"),
    ("left_shoulder", "LeftShoulder"),
    ("right_shoulder", "RightShoulder"),
    ("left_elbow", "LeftElbow"),
    ("right_elbow", "RightElbow"),
    ("left_wrist", "LeftWrist"),
    ("right_wrist", "RightWrist"),
]
