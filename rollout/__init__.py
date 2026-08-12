# Rollout module


def _get_tmaze_eval():
    from rollout.tmaze.eval_policy import eval_policy
    return eval_policy


def _get_add_eval():
    from rollout.add.eval_policy import eval_policy
    return eval_policy


def _get_scattered_copy_eval():
    from rollout.scattered_copy.eval_policy import eval_policy
    return eval_policy


def _get_ltmb_eval():
    from rollout.ltmb.eval_policy import eval_policy
    return eval_policy


def _get_mikasa_robo_eval():
    from rollout.mikasa_robo.eval_policy import eval_policy
    return eval_policy


eval_dict = {
    'tmaze': _get_tmaze_eval,
    'add': _get_add_eval,
    'scattered_copy': _get_scattered_copy_eval,
    'ltmb': _get_ltmb_eval,
    'mikasa_robo': _get_mikasa_robo_eval,
}
