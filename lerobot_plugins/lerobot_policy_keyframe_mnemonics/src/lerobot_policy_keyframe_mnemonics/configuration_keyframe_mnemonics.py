from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig

ARCHITECTURES = ("single_step_transformer", "flow_matching_dit")


@PreTrainedConfig.register_subclass("keyframe_mnemonics")
@dataclass
class KeyframeMnemonicsConfig(PreTrainedConfig):
    """Inference-time config for the Keyframe Mnemonics pipeline.

    The policy is trained in the keyframe-mnemonics repo (proxy -> selector ->
    policy); this config only describes how to rebuild the frozen selector and
    stage-3 policy so `lerobot-record` can roll them out.

    `input_features` describe the *live robot* observation (un-indexed camera
    keys). The per-slot expansion the stage-3 dataset uses happens inside
    `select_action`, driven by `total_slots`.

    The pipeline trains on the normalized space defined by the proxy H5, so the
    processors reproduce it: MEAN_STD for state, MIN_MAX for actions, and IDENTITY
    for images (which are raw [0, 1] throughout). `export.py` supplies the stats.
    """

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

    # Stage-3 policy
    architecture: str = "single_step_transformer"
    chunk_size: int = 1
    n_action_steps: int = 1
    hidden_dim: int = 1024
    attn_heads: int = 16
    num_layers: int = 4
    num_inference_steps: int = 5

    # Frozen PPO selector
    selector_features_dim: int = 128
    selector_hidden_dim: int = 256
    selector_net_arch: list[int] = field(default_factory=lambda: [500, 500])
    selector_obs_low: float = -6.283185307179586
    selector_obs_high: float = 6.283185307179586

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    def __post_init__(self):
        super().__post_init__()

        if self.architecture not in ARCHITECTURES:
            raise ValueError(
                f"`architecture` must be one of {ARCHITECTURES}. Got {self.architecture}."
            )
        if self.architecture == "single_step_transformer" and self.chunk_size != 1:
            raise ValueError(
                "single_step_transformer predicts one action; `chunk_size` must be 1. "
                f"Got {self.chunk_size}."
            )
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

    @property
    def observation_delta_indices(self) -> None:
        # History lives in the memory buffer, not in a time window of frames.
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=5e-5, weight_decay=0.05)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        camera_key = f"observation.images.{self.camera_name}"
        if camera_key not in (self.input_features or {}):
            raise ValueError(
                f"KeyframeMnemonics expects an image feature for `camera_name`. "
                f"Missing: {camera_key}. Available: {sorted(self.image_features)}"
            )
        if self.robot_state_feature is None:
            raise ValueError("KeyframeMnemonics requires 'observation.state' among the inputs.")
        if self.action_feature is None:
            raise ValueError("KeyframeMnemonics requires 'action' in output_features.")
