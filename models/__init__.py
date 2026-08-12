class _LazyDict(dict):
    """Dictionary that calls functions on access to enable lazy loading."""
    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(value, dict):
            # Recursively wrap nested dicts
            return _LazyDict({k: v for k, v in value.items()})
        elif callable(value):
            # Call the function to get the actual class
            return value()
        return value

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def _get_tmaze_selector():
    from .tmaze_model import SelectorFeatureExtractor
    return SelectorFeatureExtractor


def _get_tmaze_proxy():
    from .tmaze_model import ProxyModel
    return ProxyModel


def _get_tmaze_policy():
    from .tmaze_model import TMazePolicy
    return TMazePolicy


def _get_mikasa_robo_selector():
    from .mikasa_robo_model import SelectorFeatureExtractor
    return SelectorFeatureExtractor


def _get_mikasa_robo_proxy():
    from .mikasa_robo_model import ProxyModel
    return ProxyModel


def _get_mikasa_robo_policy():
    from .mikasa_robo_model import MikasaRoboPolicy
    return MikasaRoboPolicy


def _get_real_robot_selector():
    from .real_robot_model import SelectorFeatureExtractor
    return SelectorFeatureExtractor


def _get_real_robot_proxy():
    from .real_robot_model import ProxyModel
    return ProxyModel


def _get_real_robot_policy():
    from .real_robot_model import RealRobotPolicy
    return RealRobotPolicy


def _get_add_selector():
    from .add_model import SelectorFeatureExtractor
    return SelectorFeatureExtractor


def _get_add_proxy():
    from .add_model import ProxyModel
    return ProxyModel


def _get_add_policy():
    from .add_model import AddPolicy
    return AddPolicy


def _get_scattered_copy_selector():
    from .scattered_copy_model import SelectorFeatureExtractor
    return SelectorFeatureExtractor


def _get_scattered_copy_proxy():
    from .scattered_copy_model import ProxyModel
    return ProxyModel


def _get_scattered_copy_policy():
    from .scattered_copy_model import ScatteredCopyPolicy
    return ScatteredCopyPolicy


def _get_ltmb_selector():
    from .ltmb_model import SelectorFeatureExtractor
    return SelectorFeatureExtractor


def _get_ltmb_proxy():
    from .ltmb_model import ProxyModel
    return ProxyModel


def _get_ltmb_policy():
    from .ltmb_model import LtmbPolicy
    return LtmbPolicy


model_dict = _LazyDict(
    tmaze=dict(
        selector=_get_tmaze_selector,
        proxy=_get_tmaze_proxy,
        policy=_get_tmaze_policy,
    ),
    mikasa_robo=dict(
        selector=_get_mikasa_robo_selector,
        proxy=_get_mikasa_robo_proxy,
        policy=_get_mikasa_robo_policy,
    ),
    real_robot=dict(
        selector=_get_real_robot_selector,
        proxy=_get_real_robot_proxy,
        policy=_get_real_robot_policy,
    ),
    add=dict(
        selector=_get_add_selector,
        proxy=_get_add_proxy,
        policy=_get_add_policy,
    ),
    scattered_copy=dict(
        selector=_get_scattered_copy_selector,
        proxy=_get_scattered_copy_proxy,
        policy=_get_scattered_copy_policy,
    ),
    ltmb=dict(
        selector=_get_ltmb_selector,
        proxy=_get_ltmb_proxy,
        policy=_get_ltmb_policy,
    )
)
