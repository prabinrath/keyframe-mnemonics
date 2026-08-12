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


def _get_tmaze_problem():
    from .tmaze_problem import TMazeProblem
    return TMazeProblem


def _get_tmaze_evaluator():
    from .tmaze_problem import TMazeEvaluator
    return TMazeEvaluator


def _get_add_problem():
    from .add_problem import AddProblem
    return AddProblem


def _get_add_evaluator():
    from .add_problem import AddEvaluator
    return AddEvaluator


def _get_scattered_copy_problem():
    from .scattered_copy_problem import ScatteredCopyProblem
    return ScatteredCopyProblem


def _get_scattered_copy_evaluator():
    from .scattered_copy_problem import ScatteredCopyEvaluator
    return ScatteredCopyEvaluator


def _get_mikasa_robo_problem():
    from .mikasa_robo_problem import MikasaRoboProblem
    return MikasaRoboProblem


def _get_mikasa_robo_policy_evaluator():
    from .mikasa_robo_problem import MikasaPolicyEvaluator
    return MikasaPolicyEvaluator


def _get_ltmb_problem():
    from .ltmb_problem import LTMBProblem
    return LTMBProblem


def _get_ltmb_evaluator():
    from .ltmb_problem import LTMBEvaluator
    return LTMBEvaluator


def _get_tmaze_policy_dataset_generator():
    from .tmaze_problem.generate_policy_dataset import generate
    return generate


def _get_add_policy_dataset_generator():
    from .add_problem.generate_policy_dataset import generate
    return generate


def _get_scattered_copy_policy_dataset_generator():
    from .scattered_copy_problem.generate_policy_dataset import generate
    return generate


def _get_ltmb_policy_dataset_generator():
    from .ltmb_problem.generate_policy_dataset import generate
    return generate


def _get_mikasa_robo_policy_dataset_generator():
    from .mikasa_robo_problem.generate_policy_dataset import generate
    return generate


def _get_tmaze_policy_evaluator():
    from .tmaze_problem import TMazePolicyEvaluator
    return TMazePolicyEvaluator


def _get_add_policy_evaluator():
    from .add_problem import AddPolicyEvaluator
    return AddPolicyEvaluator


def _get_scattered_copy_policy_evaluator():
    from .scattered_copy_problem import ScatteredCopyPolicyEvaluator
    return ScatteredCopyPolicyEvaluator


def _get_ltmb_policy_evaluator():
    from .ltmb_problem import LTMBPolicyEvaluator
    return LTMBPolicyEvaluator


def _get_mikasa_robo_evaluator():
    from .mikasa_robo_problem import MikasaEvaluator
    return MikasaEvaluator


problem_dict = _LazyDict(
    tmaze=dict(
        problem=_get_tmaze_problem,
        evaluator=_get_tmaze_evaluator,
        policy_evaluator=_get_tmaze_policy_evaluator,
        policy_dataset_generator=_get_tmaze_policy_dataset_generator,
    ),
    add=dict(
        problem=_get_add_problem,
        evaluator=_get_add_evaluator,
        policy_evaluator=_get_add_policy_evaluator,
        policy_dataset_generator=_get_add_policy_dataset_generator,
    ),
    scattered_copy=dict(
        problem=_get_scattered_copy_problem,
        evaluator=_get_scattered_copy_evaluator,
        policy_evaluator=_get_scattered_copy_policy_evaluator,
        policy_dataset_generator=_get_scattered_copy_policy_dataset_generator,
    ),
    ltmb=dict(
        problem=_get_ltmb_problem,
        evaluator=_get_ltmb_evaluator,
        policy_evaluator=_get_ltmb_policy_evaluator,
        policy_dataset_generator=_get_ltmb_policy_dataset_generator,
    ),
    mikasa_robo=dict(
        problem=_get_mikasa_robo_problem,
        evaluator=_get_mikasa_robo_evaluator,
        policy_evaluator=_get_mikasa_robo_policy_evaluator,
        policy_dataset_generator=_get_mikasa_robo_policy_dataset_generator,
    )
)
