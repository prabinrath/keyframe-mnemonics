from collections import deque

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import Tensor

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import build_inference_frame
from lerobot.utils.constants import OBS_STATE

from keyframe_mnemonics.buffers import queue_dict
from models.real_robot_model import SelectorFeatureExtractor
from problems.real_robot_problem.lerobot_utils import (
    OBS_DIM,
    STATE_NAMES,
    build_vla_observation,
    flatten_observation,
    get_lerobot_features,
)

from .configuration_keyframe_buffer import KeyframeBufferConfig


def _build_selector(config: KeyframeBufferConfig) -> ActorCriticPolicy:
    """Rebuild the frozen PPO selector's network so its weights can be loaded."""
    observation_space = gym.spaces.Box(
        low=config.selector_obs_low,
        high=config.selector_obs_high,
        shape=(OBS_DIM,),
        dtype=np.float32,
    )
    action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
    return ActorCriticPolicy(
        observation_space,
        action_space,
        lr_schedule=lambda _: 0.0,
        net_arch=list(config.selector_net_arch),
        features_extractor_class=SelectorFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=config.selector_features_dim,
            hidden_dim=config.selector_hidden_dim,
        ),
    )


class KeyframeBufferPolicy(PreTrainedPolicy):
    """Selector + memory buffer in front of a LeRobot policy trained on stage-3 data.

    Only the selector's weights live in this checkpoint. The inner policy is loaded
    from `config.inner_policy_path` with its own processors, so its normalization
    and tokenization stay exactly as `lerobot-train` produced them.
    """

    config_class = KeyframeBufferConfig
    name = "keyframe_buffer"

    def __init__(self, config: KeyframeBufferConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.selector = _build_selector(config)

        # Held outside the module tree (tuple, not attribute assignment) so the
        # inner policy's weights are not duplicated into this checkpoint.
        inner = get_policy_class(config.inner_policy_type).from_pretrained(config.inner_policy_path)
        inner.config.n_action_steps = config.n_action_steps
        inner.to(config.device)
        inner.eval()
        inner_pre, inner_post = make_pre_post_processors(
            inner.config,
            pretrained_path=config.inner_policy_path,
            preprocessor_overrides={"device_processor": {"device": config.device}},
            postprocessor_overrides={"device_processor": {"device": config.device}},
        )
        self._inner = (inner, inner_pre, inner_post)

        self._ds_features = get_lerobot_features(config.total_slots)

        if config.state_mean is not None:
            self.register_buffer("state_mean", torch.tensor(config.state_mean, dtype=torch.float32))
            self.register_buffer("state_std", torch.tensor(config.state_std, dtype=torch.float32))
        else:
            self.state_mean = None
            self.state_std = None

        self._buffer = None
        self._action_queue = deque([], maxlen=config.n_action_steps)
        self.reset()

    @property
    def inner(self):
        return self._inner[0]

    def get_optim_params(self) -> dict:
        return self.parameters()

    # The selector's ResNet is held channels_last, whose strides safetensors cannot
    # flatten. Both directions go through a contiguous copy; layout is restored on load.

    def _save_pretrained(self, save_directory, state_dict=None) -> None:
        if state_dict is None:
            state_dict = self.state_dict()
        super()._save_pretrained(
            save_directory, state_dict={k: v.contiguous() for k, v in state_dict.items()}
        )

    @classmethod
    def _load_as_safetensor(cls, model, model_file, map_location, strict):
        for module in model.modules():
            for name, param in list(module._parameters.items()):
                if param is not None and not param.is_contiguous():
                    module._parameters[name] = torch.nn.Parameter(
                        param.data.contiguous(), requires_grad=param.requires_grad
                    )
            for name, buf in list(module._buffers.items()):
                if buf is not None and not buf.is_contiguous():
                    module._buffers[name] = buf.contiguous()
        model = super()._load_as_safetensor(model, model_file, map_location, strict)
        return model.to(memory_format=torch.channels_last)

    def reset(self):
        """Clear the memory buffer and the inner policy's own caches. Called per episode."""
        self._buffer = queue_dict[self.config.queue_strategy](
            maxsize=self.config.buffer_size,
            use_current_obs=True,
            data_size=(OBS_DIM,),
            rejection_threshold=self.config.rejection_threshold,
            no_repeat_threshold=self.config.no_repeat_threshold,
        )
        self._action_queue.clear()
        self.inner.reset()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError(
            "keyframe_buffer is inference-only. Train the selector with the "
            "keyframe-mnemonics pipeline and the inner policy with lerobot-train on the "
            "stage-3 policy dataset."
        )

    def _flatten(self, batch: dict[str, Tensor]) -> Tensor:
        """Build the selector's flat vector. State arrives raw (the config pins
        STATE to IDENTITY) and is normalized here to match the proxy H5."""
        state = batch[OBS_STATE]
        if state.shape[0] != 1:
            raise ValueError(
                f"keyframe_buffer keeps a per-episode memory buffer and only supports "
                f"batch size 1 at inference. Got {state.shape[0]}."
            )
        state = state[0].detach().float().cpu()
        if self.state_mean is not None:
            state = (state - self.state_mean.cpu()) / self.state_std.cpu()

        wrist = batch[f"observation.images.{self.config.camera_name}"][0].detach().float().cpu()
        return torch.from_numpy(flatten_observation(wrist, state))

    def _step_buffer(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Score the observation, update the buffer, and hand the inner policy a frame."""
        flat = self._flatten(batch)

        obs = flat.unsqueeze(0).to(self.config.device)
        priority = self.selector._predict(obs, deterministic=True).clamp(0.0, 1.0).item()
        self._buffer.push(flat, priority)

        observation = build_vla_observation(self._buffer.get().numpy(), self.config.total_slots)
        # The buffer's state is normalized, but the inner policy was trained on a
        # --no_normalization dataset. The last slot is this frame, so the raw state
        # from the batch is an exact substitute.
        raw_state = batch[OBS_STATE][0].detach().float().cpu()
        observation.update(zip(STATE_NAMES, (float(v) for v in raw_state)))

        frame = build_inference_frame(
            observation=observation,
            ds_features=self._ds_features,
            device=self.config.device,
            task=self.config.task,
        )
        _, inner_pre, _ = self._inner
        return inner_pre(frame)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        return self.inner.predict_action_chunk(self._step_buffer(batch))

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return one action. The selector runs every call; the inner policy manages
        its own chunk cache, so this delegates rather than queueing again."""
        self.eval()
        frame = self._step_buffer(batch)
        _, _, inner_post = self._inner
        return inner_post(self.inner.select_action(frame))
