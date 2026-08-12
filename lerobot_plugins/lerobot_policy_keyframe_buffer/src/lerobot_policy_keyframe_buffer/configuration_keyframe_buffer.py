from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig


@PreTrainedConfig.register_subclass("keyframe_buffer")
@dataclass
class KeyframeBufferConfig(PreTrainedConfig):
    """Wraps any LeRobot policy with the Keyframe Mnemonics selector and memory buffer.

    The inner policy is one trained by `lerobot-train` on a stage-3 policy dataset,
    which sees per-slot camera keys (`wrist_img1..N`). This wrapper
    takes the *live robot* observation, runs the frozen selector, maintains the
    buffer, and expands it into those per-slot keys — the same thing
    `rollout/mikasa_robo/eval_vla.py` does offline, moved inside a policy so
    `lerobot-record` can drive it on hardware.

    The inner policy is referenced by path rather than nested into this
    checkpoint, so its weights and processors are not duplicated and it can be
    swapped without re-exporting the selector.
    """

    # Inner LeRobot policy
    inner_policy_type: str = "act"
    inner_policy_path: str = ""

    # Memory buffer
    buffer_size: int = 2
    total_slots: int = 3
    queue_strategy: str = "evict_latest_norepeat"
    rejection_threshold: float = 0.9
    no_repeat_threshold: float = 0.1

    # Flat observation layout consumed by the selector
    image_size: int = 128
    state_dim: int = 8
    camera_name: str = "wrist_img"

    # Action chunking, mirrored onto the inner policy at load time
    chunk_size: int = 1
    n_action_steps: int = 1

    # Frozen PPO selector
    selector_features_dim: int = 128
    selector_hidden_dim: int = 256
    selector_net_arch: list[int] = field(default_factory=lambda: [500, 500])
    selector_obs_low: float = -6.283185307179586
    selector_obs_high: float = 6.283185307179586

    # Set only when the proxy H5 was built with --normalize mean_std
    state_mean: list[float] | None = None
    state_std: list[float] | None = None

    # The inner policy owns its own normalization; this wrapper passes through.
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    task: str | None = None

    def __post_init__(self):
        super().__post_init__()

        if not self.inner_policy_path:
            raise ValueError("`inner_policy_path` must point at a trained LeRobot checkpoint.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"`n_action_steps` ({self.n_action_steps}) cannot exceed `chunk_size` "
                f"({self.chunk_size})."
            )
        if self.total_slots != self.buffer_size + 1:
            raise ValueError(
                "`total_slots` must be `buffer_size` + 1 (the current observation "
                f"occupies the extra slot). Got {self.total_slots} and {self.buffer_size}."
            )
        if (self.state_mean is None) != (self.state_std is None):
            raise ValueError("`state_mean` and `state_std` must be set together.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=1e-4, weight_decay=1e-4)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        camera_key = f"observation.images.{self.camera_name}"
        if camera_key not in (self.input_features or {}):
            raise ValueError(
                f"keyframe_buffer expects an image feature for `camera_name`. "
                f"Missing: {camera_key}. Available: {sorted(self.image_features)}"
            )
        if self.robot_state_feature is None:
            raise ValueError("keyframe_buffer requires 'observation.state' among the inputs.")
