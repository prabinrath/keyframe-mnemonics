"""Keyframe Mnemonics selector + memory buffer wrapping any LeRobot policy.

Importing this package registers the policy type "keyframe_buffer", after which
LeRobot resolves the modeling and processor modules by naming convention.
"""

try:
    import lerobot  # noqa: F401
except ImportError as e:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    ) from e

from .configuration_keyframe_buffer import KeyframeBufferConfig

__all__ = ["KeyframeBufferConfig"]
