"""LeRobot dataset utilities for mikasa_robo problem."""

import numpy as np


def get_lerobot_features(buffer_size):
    """Get LeRobot dataset features for mikasa_robo problem.

    Args:
        buffer_size: Total number of observations (historical + current context)

    Returns:
        Dictionary of features for LeRobot dataset
    """
    features = {}

    # For each buffer: overhead_camera, gripper_camera
    for i in range(1, buffer_size + 1):
        features[f"observation.images.overhead_camera{i}"] = {
            "dtype": "video",
            "shape": (128, 128, 3),
            "names": ("height", "width", "channels"),
        }
        features[f"observation.images.gripper_camera{i}"] = {
            "dtype": "video",
            "shape": (128, 128, 3),
            "names": ("height", "width", "channels"),
        }

    features["observation.state"] = {
        "dtype": "float32",
        "shape": (9,),
        "names": [f"qpos_{j}" for j in range(9)],
    }
    
    features["action"] = {
        "dtype": "float32",
        "shape": (8,),
        "names": [f"action_{j}" for j in range(8)],
    }
    
    return features


def build_policy_observation(buffer, buffer_size):
    """Build an observation dict formatted for policy inference.

    Args:
        buffer: Flattened buffer array
        buffer_size: Total number of observations (historical + current context)

    Returns:
        Dictionary with parsed observations for each buffer
    """
    obs_dim = 98329  # overhead_camera (49152) + gripper_camera (49152) + tcp_pose (7) + qpos (9) + qvel (9)
    frame_data = {}

    # Split buffer into individual observations and extract images from all buffers
    for i in range(buffer_size):
        start_idx = i * obs_dim
        obs_data = buffer[start_idx:start_idx + obs_dim]

        # Extract and reshape components based on process_env_observation structure
        # overhead_camera (49152), gripper_camera (49152), tcp_pose (7), qpos (9), qvel (9)
        overhead_camera = obs_data[0:49152].reshape(128, 128, 3)
        overhead_camera = (overhead_camera * 255).round().astype(np.uint8)
        frame_data[f"observation.images.overhead_camera{i+1}"] = overhead_camera

        gripper_camera = obs_data[49152:98304].reshape(128, 128, 3)
        gripper_camera = (gripper_camera * 255).round().astype(np.uint8)
        frame_data[f"observation.images.gripper_camera{i+1}"] = gripper_camera
    
    # Extract qpos only from the last buffer element (joints[7:16], offset 49152+49152+7=98311)
    last_idx = (buffer_size - 1) * obs_dim
    last_obs_data = buffer[last_idx:last_idx + obs_dim]
    frame_data["observation.state"] = last_obs_data[98311:98320].astype(np.float32)
    
    return frame_data


def build_vla_observation(buffer, buffer_size):
    """Build an observation dict formatted for VLA inference via build_inference_frame.

    Produces the exact key/value format that build_inference_frame (via
    build_dataset_frame) expects:
      - Image keys use the short name after stripping the "observation.images."
        prefix, e.g. "gripper_camera1", "overhead_camera1".
        Values are uint8 HxWxC numpy arrays.
      - State keys match the "names" entries in the features spec,
        e.g. "qpos_0" ... "qpos_8". Values are Python floats.

    Args:
        buffer: Flattened buffer numpy array.
        buffer_size: Total number of slots (historical + current context).

    Returns:
        Dictionary ready to pass as ``observation`` to build_inference_frame.
    """
    obs_dim = 98329  # overhead_camera (49152) + gripper_camera (49152) + tcp_pose (7) + qpos (9) + qvel (9)
    observation = {}

    for i in range(buffer_size):
        start_idx = i * obs_dim
        obs_data = buffer[start_idx:start_idx + obs_dim]

        overhead = obs_data[0:49152].reshape(128, 128, 3)
        observation[f"overhead_camera{i + 1}"] = (overhead * 255).round().astype(np.uint8)

        gripper = obs_data[49152:98304].reshape(128, 128, 3)
        observation[f"gripper_camera{i + 1}"] = (gripper * 255).round().astype(np.uint8)

    # State: qpos from the last (most-recent) buffer slot
    last_obs = buffer[(buffer_size - 1) * obs_dim:(buffer_size - 1) * obs_dim + obs_dim]
    qpos = last_obs[98311:98320].astype(np.float32)
    for j in range(9):
        observation[f"qpos_{j}"] = float(qpos[j])

    return observation