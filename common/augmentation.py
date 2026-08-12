import re

import torch
from torchvision.transforms import v2 as T


class RandomCurrentObsBlackout:
    """Zeros the current-observation slot with probability p, independently per sample.

    The buffer holds the current observation in the last slot, so this trains the
    policy to fall back on the selector's keyframes.
    """

    def __init__(self, p=0.0):
        self.p = p
        self.slot_re = re.compile(r'(\d+)$')

    def slot_index(self, key):
        match = self.slot_re.search(key)
        return int(match.group(1)) if match else -1

    def __call__(self, observations):
        if self.p <= 0:
            return observations

        keys = [k for k in observations if k.startswith('observation.images')]
        if not keys:
            return observations

        last = max(self.slot_index(k) for k in keys)
        current = [k for k in keys if self.slot_index(k) == last]

        slot = observations[current[0]]
        # One mask across the slot's cameras, so the whole observation drops together
        mask = torch.rand(slot.shape[0], device=slot.device) < self.p
        for key in current:
            observations[key][mask] = 0.0
        return observations


class ImageAugmenter:
    """Applies random augmentations to all image observations in-place.
    """

    def __init__(self):
        self.blackout = RandomCurrentObsBlackout(p=0.3)
        self.transform = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=[0.8, 1.2])], p=0.5),
            T.RandomApply([T.ColorJitter(contrast=[0.8, 1.2])], p=0.5),
            T.RandomApply([T.ColorJitter(saturation=[0.5, 1.5])], p=0.5),
            T.RandomApply([T.ColorJitter(hue=[-0.05, 0.05])], p=0.5),
            T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=1.5)], p=0.5),
            T.RandomApply([T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))], p=0.5),
        ])

    def __call__(self, observations):
        observations = self.blackout(observations)
        for key in observations:
            if key.startswith('observation.images'):
                observations[key] = self.transform(observations[key])
        return observations
