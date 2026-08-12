from problems.problem import Problem, Evaluator
from common.helpers import parse_experiment_name
import wandb
import random
import numpy as np
from pathlib import Path
import os


class AddProblem(Problem):
    def __init__(
        self,
        min_len,
        max_len,
        min_val=0.0,
        max_val=1.0,
        **kwargs,
    ):
        if min_len < 3:
            raise ValueError("min_len must be at least 3.")
        if min_len > max_len:
            raise ValueError("min_len must be <= max_len.")

        self.min_len = int(min_len)
        self.max_len = int(max_len)
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.current_target = 0.0
        self.current_obs = None

        kwargs.setdefault("seq_len", self.max_len)
        kwargs.setdefault("buffer_size", 2)
        super().__init__(
            observation_shape=(3,),
            observation_type=float,
            **kwargs,
        )

        self.reset()

    def _build_variation(self):
        seq_len = random.randint(self.min_len, self.max_len)
        emit_idx = seq_len - 1

        first_idx = random.randint(0, emit_idx - 2)
        second_idx = random.randint(first_idx + 1, emit_idx - 1)

        values = np.random.uniform(self.min_val, self.max_val, size=(seq_len,)).astype(np.float32)
        observations = np.zeros((seq_len, 3), dtype=np.float32)
        observations[:, 0] = values

        observations[first_idx, 1] = 1.0
        observations[second_idx, 2] = 1.0
        observations[emit_idx] = np.asarray([0.0, 1.0, 1.0], dtype=np.float32)

        target_sum = float(values[first_idx] + values[second_idx])
        return {
            "seq_len": seq_len,
            "observations": observations,
            "target_sum": target_sum,
        }

    def get_sample(self, idx):
        variation = self.samples[self.sidx]
        self.current_obs = variation["observations"][idx]
        if idx == self.seq_len - 1:
            self.current_target = variation["target_sum"]
        else:
            self.current_target = 0.0
        return self.current_obs

    def get_reward(self):
        # Sparse reward: 1.0 at the emit position (last step), 0 elsewhere
        if self.current_obs is not None and self.current_obs[1] == 1.0 and self.current_obs[2] == 1.0:
            return 1.0
        return 0.0

    def get_target(self, t):
        return np.asarray([self.current_target], dtype=np.float32)

    def reset(self):
        if self.samples is None:
            self.samples = [self._build_variation() for _ in range(self.num_variations)]
            self.sidx = 0
        else:
            self.sidx = random.randint(0, self.num_variations - 1) if self.randomize_reset \
                else (self.sidx + 1) % self.num_variations

        variation = self.samples[self.sidx]
        self.seq_len = variation["seq_len"]
        self.current_target = 0.0
        self.current_obs = None

    def reset_sidx(self, sidx):
        self.sidx = sidx
        variation = self.samples[self.sidx]
        self.seq_len = variation["seq_len"]
        self.current_target = 0.0
        self.current_obs = None


class AddEvaluator(Evaluator):
    def __init__(self, loss_threshold, evaluation_rollouts, path_prefix="", **kwargs):
        super().__init__(**kwargs)
        self.loss_threshold = loss_threshold
        self.evaluation_rollouts = evaluation_rollouts
        self.checkpoint_path = os.path.join(path_prefix, f"checkpoints/{self.checkpoint_folder}")
        Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)
        self.experiment_name = parse_experiment_name(self.checkpoint_folder)

    def evaluate(self, selector, proxy, train_process, test_process):
        print("\n------------ID Evaluation------------")
        loss = self.evaluate_end_to_end(selector, proxy, train_process)
        print("------------OOD Evaluation------------")
        self.evaluate_end_to_end(selector, proxy, test_process)
        print("--------------------------------------\n")
        return loss < self.loss_threshold

    def evaluate_end_to_end(self, selector, proxy, process):
        process.problem.randomize_reset = True
        first_marked_priority = []
        second_marked_priority = []
        irrelevant_priority = []
        emit_priority = []
        final_mse_episode = []

        for _ in range(self.evaluation_rollouts):
            process.reset()
            for idx in range(process.problem.seq_len):
                smp, t = process.get_obs(idx)
                target = process.get_target(t)
                p, _ = selector.get_priority(smp)
                process.set_action(smp, p)

                if smp[1] == 1 and smp[2] == 1:
                    emit_priority.append(p)
                elif smp[1] == 1:
                    first_marked_priority.append(p)
                elif smp[2] == 1:
                    second_marked_priority.append(p)
                else:
                    irrelevant_priority.append(p)

                pred_target = proxy.get_action(
                    process.get_buffer().unsqueeze(0).to(proxy.device)
                ).cpu().squeeze()
                if smp[1] == 1 and smp[2] == 1:
                    final_mse_episode.append(((target - pred_target) ** 2).mean())

        mean_final_mse = sum(final_mse_episode) / len(final_mse_episode)
        print(f"terminal - buffer state: {process.get_buffer()} | target: {target} | predicted: {pred_target}")
        if first_marked_priority:
            print(f"avg first-marked priority: {sum(first_marked_priority)/len(first_marked_priority)}")
        if second_marked_priority:
            print(f"avg second-marked priority: {sum(second_marked_priority)/len(second_marked_priority)}")
        if irrelevant_priority:
            print(f"avg irrelevant priority: {sum(irrelevant_priority)/len(irrelevant_priority)}")
        if emit_priority:
            print(f"avg emit priority: {sum(emit_priority)/len(emit_priority)}")
        print(f"mean final mse: {mean_final_mse}")
        return mean_final_mse


class AddPolicyEvaluator(Evaluator):
    """Policy-stage evaluator: selector+policy eval via the rollout registry in an
    isolated subprocess; early-stops when the final MSE drops below loss_threshold
    (lower is better)."""

    def __init__(self, loss_threshold, logging, evaluation_rollouts, **kwargs):
        super().__init__(**kwargs)
        self.loss_threshold = loss_threshold
        self.logging = logging
        self.evaluation_rollouts = evaluation_rollouts
        if self.logging:
            wandb.define_metric("eval_step")
            wandb.define_metric("eval/*", step_metric="eval_step")
            self.step = 1

    def evaluate(self, eval_fn, policy_checkpoint, checkpoint_num):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
            result = ex.submit(
                eval_fn,
                policy_checkpoint=policy_checkpoint,
                checkpoint_num=str(checkpoint_num),
                episode_indices=list(range(self.evaluation_rollouts)),
            ).result()
        mse = result
        print(f"Policy Eval — final MSE: {mse:.6f}")
        if self.logging:
            wandb.log({"eval/policy_mse": mse, "eval_step": self.step})
            self.step += 1
        return mse < self.loss_threshold
