from torchvision.transforms import v2 as T


class ImageAugmenter:
    """Applies random augmentations to all image observations in-place.
    """

    def __init__(self):
        self.transform = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=[0.8, 1.2])], p=0.5),
            T.RandomApply([T.ColorJitter(contrast=[0.8, 1.2])], p=0.5),
            T.RandomApply([T.ColorJitter(saturation=[0.5, 1.5])], p=0.5),
            T.RandomApply([T.ColorJitter(hue=[-0.05, 0.05])], p=0.5),
            T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=1.5)], p=0.5),
            T.RandomApply([T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))], p=0.5),
        ])

    def __call__(self, observations):
        for key in observations:
            if key.startswith('observation.images'):
                observations[key] = self.transform(observations[key])
        return observations