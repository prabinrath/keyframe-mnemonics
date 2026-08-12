from typing import Any

import torch

from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    make_default_pre_post_processors,
)

from .configuration_keyframe_mnemonics import KeyframeMnemonicsConfig


def make_keyframe_mnemonics_pre_post_processors(
    config: KeyframeMnemonicsConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Standard scaffold pipelines. Normalization is IDENTITY (see the config docstring),
    so these only batch, move to device, and move actions back to CPU."""
    return make_default_pre_post_processors(config, dataset_stats, normalizer_device=config.device)
