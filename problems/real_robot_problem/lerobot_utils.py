"""LeRobot dataset utilities for the real_robot problem.

Flat observation layout consumed by the selector and proxy:

    wrist_img (128*128*3) + state (8) = 49160

Images are stored HWC in [0, 1]; the models reshape and permute to CHW themselves.

Single source of truth for that layout: the stage-1 converter, the stage-3
generator and the inference-time LeRobot plugin all build observations through
here so the resize can never drift between training and rollout.

The dataset and the robot also carry a front camera; nothing here reads it.
"""

import numpy as np
import torch
from torchvision.transforms.v2.functional import resize

IMAGE_SIZE = 128
STATE_DIM = 8
ACTION_DIM = 8
PIXELS_PER_CAMERA = IMAGE_SIZE * IMAGE_SIZE * 3
OBS_DIM = PIXELS_PER_CAMERA + STATE_DIM

# `record_dataset.py` captured at CAMERA_CAPTURE_HW and wrote RECORD_IMAGE_SIZE
# via cv2.INTER_LINEAR — bilinear, no antialiasing.
RECORD_IMAGE_SIZE = 224
CAMERA_CAPTURE_HW = (480, 640)

# Match the collected dataset's key so the stage-3 dataset stays readable
# as a standalone LeRobot dataset.
CAMERA_NAME = "wrist_img"
STATE_NAMES = [f"joint{j}" for j in range(1, 8)] + ["gripper"]
ACTION_NAMES = list(STATE_NAMES)


def resize_camera(image):
    """Resize a CHW float image in [0, 1] to IMAGE_SIZE x IMAGE_SIZE.

    A live frame takes the same two legs the training frames did, so it arrives
    with the same aliasing. Dataset frames are already 224 and skip the first.
    """
    height, width = image.shape[-2:]
    if height > RECORD_IMAGE_SIZE or width > RECORD_IMAGE_SIZE:
        image = resize(image, [RECORD_IMAGE_SIZE, RECORD_IMAGE_SIZE], antialias=False)
    if image.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
        image = resize(image, [IMAGE_SIZE, IMAGE_SIZE], antialias=True)
    return image.to(torch.float32).clamp_(0.0, 1.0)


def flatten_observation(wrist_img, state):
    """Build one flat observation vector from a decoded LeRobot frame."""
    camera = resize_camera(wrist_img).permute(1, 2, 0).reshape(-1)
    state = torch.as_tensor(state, dtype=torch.float32).reshape(-1)
    if state.shape[0] != STATE_DIM:
        raise ValueError(f"Expected a {STATE_DIM}-dim state, got {tuple(state.shape)}")
    return torch.cat([camera, state]).numpy().astype(np.float32)


def get_lerobot_features(total_slots):
    """Features for the stage-3 policy dataset: one video stream per slot."""
    features = {}

    for i in range(1, total_slots + 1):
        features[f"observation.images.{CAMERA_NAME}{i}"] = {
            "dtype": "video",
            "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
            "names": ("height", "width", "channels"),
        }

    features["observation.state"] = {
        "dtype": "float32",
        "shape": (STATE_DIM,),
        "names": list(STATE_NAMES),
    }

    features["action"] = {
        "dtype": "float32",
        "shape": (ACTION_DIM,),
        "names": list(ACTION_NAMES),
    }

    return features


def split_buffer(buffer, total_slots):
    """Yield (wrist, state) per slot; image uint8 HWC, state float32."""
    for i in range(total_slots):
        obs_data = buffer[i * OBS_DIM:(i + 1) * OBS_DIM]

        wrist = obs_data[:PIXELS_PER_CAMERA].reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
        wrist = (wrist * 255).round().astype(np.uint8)

        state = obs_data[PIXELS_PER_CAMERA:].astype(np.float32)
        yield wrist, state


def build_policy_observation(buffer, total_slots):
    """Parse a flat buffer into policy observation keys. State comes from the most recent slot."""
    frame_data = {}
    state = None

    for i, (wrist, slot_state) in enumerate(split_buffer(buffer, total_slots), start=1):
        frame_data[f"observation.images.{CAMERA_NAME}{i}"] = wrist
        state = slot_state

    frame_data["observation.state"] = state
    return frame_data


def build_vla_observation(buffer, total_slots):
    """Parse a flat buffer into the key format build_inference_frame expects.

    Image keys drop the "observation.images." prefix; state keys are the
    per-dimension names from the features spec.
    """
    observation = {}
    state = None

    for i, (wrist, slot_state) in enumerate(split_buffer(buffer, total_slots), start=1):
        observation[f"{CAMERA_NAME}{i}"] = wrist
        state = slot_state

    for name, value in zip(STATE_NAMES, state):
        observation[name] = float(value)

    return observation
