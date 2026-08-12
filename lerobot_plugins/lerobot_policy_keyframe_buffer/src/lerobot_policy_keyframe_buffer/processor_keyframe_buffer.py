from typing import Any

import torch

from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    make_default_pre_post_processors,
)

from .configuration_keyframe_buffer import KeyframeBufferConfig


def make_keyframe_buffer_pre_post_processors(
    config: KeyframeBufferConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Pass-through scaffold. The inner policy runs its own pre/post processors on the
    expanded per-slot frame, so normalization here is IDENTITY."""
    return make_default_pre_post_processors(config, dataset_stats, normalizer_device=config.device)
