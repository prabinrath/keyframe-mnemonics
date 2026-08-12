"""Keyframe Mnemonics as a LeRobot policy plugin.

Importing this package registers the policy type "keyframe_mnemonics", after which
LeRobot resolves the modeling and processor modules by naming convention.
"""

try:
    import lerobot  # noqa: F401
except ImportError as e:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    ) from e

from .configuration_keyframe_mnemonics import KeyframeMnemonicsConfig

__all__ = ["KeyframeMnemonicsConfig"]
