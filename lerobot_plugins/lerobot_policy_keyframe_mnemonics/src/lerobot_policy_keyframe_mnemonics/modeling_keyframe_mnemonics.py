from collections import deque

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_STATE

from keyframe_mnemonics.buffers import queue_dict
from models.real_robot_model import RealRobotPolicy, SelectorFeatureExtractor
from problems.real_robot_problem.lerobot_utils import OBS_DIM, flatten_observation

from .configuration_keyframe_mnemonics import KeyframeMnemonicsConfig


def _build_selector(config: KeyframeMnemonicsConfig) -> ActorCriticPolicy:
    """Rebuild the frozen PPO selector's network so its weights can be loaded.

    Mirrors how Selector constructs PPO in keyframe_mnemonics/selector.py: same
    feature extractor, same net_arch, and the Box(0, 1) priority action space
    used by VectorizedSelectorEnv.
    """
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


class KeyframeMnemonicsPolicy(PreTrainedPolicy):
    """Selector + memory buffer + stage-3 policy, as one LeRobot policy.

    At every step the selector scores the incoming observation and the buffer
    decides whether to keep it; the stage-3 policy then acts on the buffer's
    contents. The buffer is per-episode state cleared by `reset()`.

    The selector must see *every* timestep. `select_action` therefore pushes on
    each call and only invokes the policy network when the action queue drains,
    which is what `lerobot-record`'s synchronous loop provides. The async
    inference server drives `predict_action_chunk` directly and skips
    observations between chunks, which would starve the buffer.
    """

    config_class = KeyframeMnemonicsConfig
    name = "keyframe_mnemonics"

    def __init__(self, config: KeyframeMnemonicsConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.selector = _build_selector(config)
        self.policy = RealRobotPolicy(
            policy_type=config.architecture,
            output_dim=config.action_feature.shape[0],
            total_slots=config.total_slots,
            hidden_dim=config.hidden_dim,
            attn_heads=config.attn_heads,
            num_layers=config.num_layers,
            horizon=config.chunk_size,
            state_dim=config.state_dim,
            image_size=config.image_size,
            camera_names=(config.camera_name,),
            **({"num_inference_steps": config.num_inference_steps}
               if config.architecture == "flow_matching_dit" else {}),
        )

        self._buffer = None
        self._action_queue = deque([], maxlen=config.n_action_steps)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.parameters()

    # The ResNet backbones are held channels_last, whose strides safetensors cannot
    # flatten (it calls .view(-1) on every tensor). Both directions therefore go
    # through a contiguous copy; the layout is restored after loading so inference
    # uses the same conv kernels as training.

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
        """Clear the memory buffer and cached actions. Called per episode."""
        self._buffer = queue_dict[self.config.queue_strategy](
            maxsize=self.config.buffer_size,
            use_current_obs=True,
            data_size=(OBS_DIM,),
            rejection_threshold=self.config.rejection_threshold,
            no_repeat_threshold=self.config.no_repeat_threshold,
        )
        self._action_queue.clear()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError(
            "keyframe_mnemonics is inference-only inside LeRobot. Train it with the "
            "three-stage pipeline: python -m keyframe_mnemonics.train --experiment_name <name>, "
            "then export the run with lerobot_policy_keyframe_mnemonics.export."
        )

    def _flatten(self, batch: dict[str, Tensor]) -> Tensor:
        """Collapse a live robot observation into the selector's flat vector.

        State arrives already normalized by the preprocessor (STATE: MEAN_STD).
        """
        state = batch[OBS_STATE]
        if state.shape[0] != 1:
            raise ValueError(
                f"keyframe_mnemonics keeps a per-episode memory buffer and only supports "
                f"batch size 1 at inference. Got {state.shape[0]}."
            )
        state = state[0].detach().float().cpu()

        wrist = batch[f"observation.images.{self.config.camera_name}"][0].detach().float().cpu()
        return torch.from_numpy(flatten_observation(wrist, state))

    def _step_buffer(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Score the observation, update the buffer, and expand it for the policy."""
        from problems.real_robot_problem.lerobot_utils import build_policy_observation

        flat = self._flatten(batch)

        obs = flat.unsqueeze(0).to(self.config.device)
        priority = self.selector._predict(obs, deterministic=True)
        priority = priority.clamp(0.0, 1.0).item()

        self._buffer.push(flat, priority)

        policy_obs = build_policy_observation(self._buffer.get().numpy(), self.config.total_slots)
        out = {}
        for key, value in policy_obs.items():
            tensor = torch.from_numpy(value)
            if "image" in key:
                tensor = tensor.float().div(255.0).permute(2, 0, 1)
            out[key] = tensor.unsqueeze(0).to(self.config.device)
        return out

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Advance the buffer by one observation and return a full action chunk."""
        self.eval()
        return self.policy.get_action(self._step_buffer(batch), deterministic=True)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return one action, refilling the chunk only when the queue drains."""
        self.eval()

        # The selector runs on every call, even while cached actions remain.
        policy_obs = self._step_buffer(batch)

        if len(self._action_queue) == 0:
            actions = self.policy.get_action(policy_obs, deterministic=True)
            self._action_queue.extend(
                actions[:, : self.config.n_action_steps].transpose(0, 1)
            )
        return self._action_queue.popleft()
