import torch

ANIMAL_TYPES_V2 = ['horse_v2']

HORSE_LANDMARKS = {
    "LeftEye": 0,
    "RightEye": 1,
    "Nose": 2,
    "Neck": 3,
    "Tail": 4,
    "LeftShoulder": 5,
    "LeftElbow": 6,
    "LeftFrontHoof": 7,
    "RightShoulder": 8,
    "RightElbow": 9,
    "RightFrontHoof": 10,
    "LeftHip": 11,
    "LeftKnee": 12,
    "LeftBackHoof": 13,
    "RightHip": 14,
    "RightKnee": 15,
    "RightBackHoof": 16
}

HORSE_SKELETON_NAMES = [
    ["Neck", "LeftEye", "Nose", "RightEye", "Neck"],
    ["Neck", "LeftShoulder", "LeftElbow", "LeftFrontHoof"],
    ["Neck", "RightShoulder", "RightElbow", "RightFrontHoof"],
    ["Neck", "Tail"],
    ["Tail", "LeftHip", "LeftKnee", "LeftBackHoof"],
    ["Tail", "RightHip", "RightKnee", "RightBackHoof"]
]

HORSE_SKELETON = [[HORSE_LANDMARKS[joint] for joint in chain] for chain in HORSE_SKELETON_NAMES]

ANIMAL_SKELETONS = {'horse_v2': HORSE_SKELETON}

def mirror_landmark(landmark):
    if 'left' in landmark:
        return landmark.replace('left', 'right')
    elif 'right' in landmark:
        return landmark.replace('right', 'left')
    elif 'Left' in landmark:
        return landmark.replace('Left', 'Right')
    elif 'Right' in landmark:
        return landmark.replace('Right', 'Left')
    else:
        return landmark

HORSE_MIRROR_LANDMARKS_INDICES = [0] * len(HORSE_LANDMARKS)
for landmark, index in HORSE_LANDMARKS.items():
    HORSE_MIRROR_LANDMARKS_INDICES[index] = HORSE_LANDMARKS[mirror_landmark(landmark)]

MIRROR_LANDMARKS_INDICES = {'horse_v2': HORSE_MIRROR_LANDMARKS_INDICES}