import torch.nn as nn


class NoInitWrapper(nn.Module):
    """Wrapper that prevents weight initialization from being applied to wrapped module."""

    def __init__(self, module):
        super().__init__()
        self._module = module

    def forward(self, *args, **kwargs):
        return self._module(*args, **kwargs)

    def apply(self, fn):
        # Override apply to prevent initialization functions from traversing into wrapped module
        # Only apply the function to the wrapper itself, not the wrapped module
        fn(self)
        return self
