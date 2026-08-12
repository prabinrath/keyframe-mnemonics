from problems.problem import Problem, Evaluator
from common.helpers import parse_experiment_name
import wandb
import random
import numpy as np
from pathlib import Path
import os


class ScatteredCopyProblem(Problem):
    def __init__(
        self,
        relevant_len,
        noise_len,
        num_symbols,
        noise_symbols,
        target_blank_len=0,
        scatter_source=True,
        scatter_target=False,
        **kwargs,
    ):
        if relevant_len < 1:
            raise ValueError("relevant_len must be positive.")
        if noise_len < 0:
            raise ValueError("noise_len must be non-negative.")
        if target_blank_len < 0:
            raise ValueError("target_blank_len must be non-negative.")
        if num_symbols < 1 or noise_symbols < 1:
            raise ValueError("num_symbols and noise_symbols must be positive.")

        self.relevant_len = int(relevant_len)
        self.noise_len = int(noise_len)
        self.target_blank_len = int(target_blank_len)
        self.num_symbols = int(num_symbols)
        self.noise_symbols = int(noise_symbols)
        self.scatter_source = bool(scatter_source)
        self.scatter_target = bool(scatter_target)

        self.relevant_start = 1
        self.relevant_end = self.num_symbols
        self.noise_start = self.relevant_end + 1
        self.noise_end = self.noise_start + self.noise_symbols - 1
        self.blank_token = self.noise_end + 1
        self.delimiter_token = self.blank_token + 1

        self.current_token = None
        self.current_target_value = 0
        self.recall_start = self.relevant_len + self.noise_len

        recall_len = self.relevant_len + (self.target_blank_len if self.scatter_target else 0)
        kwargs.setdefault("seq_len", self.recall_start + recall_len)
        super().__init__(
            observation_shape=(1,),
            observation_type=int,
            **kwargs,
        )

        self.min_val = 0
        self.max_val = self.delimiter_token
        self.reset()

    def _build_variation(self):
        source_len = self.relevant_len + self.noise_len
        recall_len = self.relevant_len + (self.target_blank_len if self.scatter_target else 0)
        seq_len = source_len + recall_len

        relevant_tokens = np.random.randint(
            self.relevant_start,
            self.relevant_end + 1,
            size=(self.relevant_len,),
        ).astype(np.int64)
        observations = np.zeros((seq_len, 1), dtype=np.int64)
        source_positions = np.arange(source_len)

        if self.scatter_source:
            selected_positions = np.sort(
                np.random.choice(source_positions, size=self.relevant_len, replace=False)
            )
        else:
            selected_positions = np.arange(self.relevant_len)

        noise_positions = np.setdiff1d(source_positions, selected_positions, assume_unique=True)
        if len(noise_positions) > 0:
            noise_tokens = np.random.randint(
                self.noise_start,
                self.noise_end + 1,
                size=(len(noise_positions),),
            ).astype(np.int64)
            observations[noise_positions, 0] = noise_tokens
        observations[selected_positions, 0] = relevant_tokens

        recall_positions = np.arange(source_len, seq_len)
        if self.scatter_target:
            emit_positions = np.sort(
                np.random.choice(recall_positions, size=self.relevant_len, replace=False)
            )
            observations[recall_positions, 0] = self.blank_token
            observations[emit_positions, 0] = self.delimiter_token
        else:
            emit_positions = recall_positions
            observations[emit_positions, 0] = self.delimiter_token

        return {
            "seq_len": seq_len,
            "observations": observations,
            "relevant_tokens": relevant_tokens,
            "emit_positions": emit_positions,
        }

    def get_sample(self, idx):
        variation = self.samples[self.sidx]
        token = int(variation["observations"][idx, 0])
        self.current_token = token

        if token == self.delimiter_token:
            emit_offset = int(np.where(variation["emit_positions"] == idx)[0][0])
            self.current_target_value = int(variation["relevant_tokens"][emit_offset])
        else:
            self.current_target_value = 0
        return np.asarray([token])

    def get_reward(self):
        # Sparse reward: 1/relevant_len at each delimiter emit position, 0 elsewhere
        if self.current_token == self.delimiter_token:
            return 1.0 / self.relevant_len
        return 0.0

    def get_target(self, t):
        return np.asarray([self.current_target_value], dtype=np.float32)

    def reset(self):
        self.current_token = None
        self.current_target_value = 0

        if self.samples is None:
            self.samples = [self._build_variation() for _ in range(self.num_variations)]
            self.sidx = 0
        else:
            self.sidx = random.randint(0, self.num_variations - 1) if self.randomize_reset \
                else (self.sidx + 1) % self.num_variations

        variation = self.samples[self.sidx]
        self.seq_len = variation["seq_len"]

    def reset_sidx(self, sidx):
        self.current_token = None
        self.current_target_value = 0
        self.sidx = sidx
        variation = self.samples[self.sidx]
        self.seq_len = variation["seq_len"]


class ScatteredCopyEvaluator(Evaluator):
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
        process.problem.randomize_reset = False
        relevant_priority = []
        noise_priority = []
        delimiter_priority = []
        blank_priority = []
        step_match = []
        delimiter_match = []

        for _ in range(self.evaluation_rollouts):
            process.reset()
            for idx in range(process.problem.seq_len):
                smp, t = process.get_obs(idx)
                target = process.get_target(t)
                p, _ = selector.get_priority(smp)
                process.set_action(smp, p)

                token = int(smp.item())
                if token == process.problem.delimiter_token:
                    delimiter_priority.append(p)
                elif token == process.problem.blank_token:
                    blank_priority.append(p)
                elif process.problem.relevant_start <= token <= process.problem.relevant_end:
                    relevant_priority.append(p)
                elif process.problem.noise_start <= token <= process.problem.noise_end:
                    noise_priority.append(p)

                pred_target = proxy.get_action(
                    process.get_buffer().unsqueeze(0).to(proxy.device)
                ).cpu().squeeze()
                is_match = int(pred_target.item()) == int(target.item())
                step_match.append(float(is_match))
                if token == process.problem.delimiter_token:
                    delimiter_match.append(float(is_match))

        step_accuracy = sum(step_match) / len(step_match)
        delimiter_accuracy = sum(delimiter_match) / len(delimiter_match)
        delimiter_error = 1.0 - delimiter_accuracy
        print(f"terminal - buffer state: {process.get_buffer()} | target: {target} | predicted: {pred_target}")
        if relevant_priority:
            print(f"avg relevant priority: {sum(relevant_priority)/len(relevant_priority)}")
        if noise_priority:
            print(f"avg noise priority: {sum(noise_priority)/len(noise_priority)}")
        if delimiter_priority:
            print(f"avg delimiter priority: {sum(delimiter_priority)/len(delimiter_priority)}")
        if blank_priority:
            print(f"avg blank priority: {sum(blank_priority)/len(blank_priority)}")
        print(f"step accuracy: {step_accuracy}")
        print(f"delimiter accuracy: {delimiter_accuracy}")
        return delimiter_error


class ScatteredCopyPolicyEvaluator(Evaluator):
    """Policy-stage evaluator: selector+policy eval via the rollout registry in an
    isolated subprocess; early-stops on step accuracy."""

    def __init__(self, sr_threshold, logging, evaluation_rollouts, **kwargs):
        super().__init__(**kwargs)
        self.sr_threshold = sr_threshold
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
        acc = result
        print(f"Policy Eval — step accuracy: {acc:.3f}")
        if self.logging:
            wandb.log({"eval/policy_acc": acc, "eval_step": self.step})
            self.step += 1
        return acc > self.sr_threshold
