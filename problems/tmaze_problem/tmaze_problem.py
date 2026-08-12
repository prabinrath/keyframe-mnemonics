import numpy as np
import gymnasium as gym
import os
import random
from pathlib import Path
from problems.problem import Problem, Evaluator
from common.helpers import parse_experiment_name
import wandb

"""
Adopted from https://github.com/twni2016/Memory-RL/blob/main/envs/tmaze.py

T-Maze: originated from (Bakker, 2001) and earlier neuroscience work, 
    and here extended to unit-test several key challenges in RL:
- Exploration
- Memory and credit assignment
- Discounting and distraction
- Generalization

Finite horizon problem: episode_length
Has a corridor of corridor_length
Looks like
                        g1
o--s---------------------j
                        g2
o is the oracle point, (x, y) = (0, 0)
s is starting point, (x, y) = (o, 0)
j is T-juncation, (x, y) = (o + corridor_length, 0)
g1 is goal candidate, (x, y) = (o + corridor_length, 1)
g2 is goal candidate, (x, y) = (o + corridor_length, -1)
"""


class TMazeBase(gym.Env):
    def __init__(
        self,
        episode_length: int = 11,
        corridor_length: int = 10,
        oracle_length: int = 0,
        goal_reward: float = 1.0,
        penalty: float = 0.0,
        distract_reward: float = 0.0,
        ambiguous_position: bool = False,
        expose_goal: bool = False,
        add_timestep: bool = False,
        deterministic: bool = True # deterministic goal sampling
    ):
        """
        The Base class of TMaze, decouples episode_length and corridor_length

        Other variants:
            (Osband, 2016): distract_reward = eps > 0, goal_reward is given at T-junction (no choice).
                This only tests the exploration and discounting of agent, no memory required
            (Osband, 2020): ambiguous_position = True, add_timestep = True, supervised = True.
                This only tests the memory of agent, no exploration required (not implemented here)
        """
        super().__init__()
        assert corridor_length >= 1 and episode_length >= 1
        assert penalty <= 0.0

        self.episode_length = episode_length
        self.corridor_length = corridor_length
        self.oracle_length = oracle_length

        self.goal_reward = goal_reward
        self.penalty = penalty
        self.distract_reward = distract_reward

        self.ambiguous_position = ambiguous_position
        self.expose_goal = expose_goal
        self.add_timestep = add_timestep

        self.action_space = gym.spaces.Discrete(4)  # four directions
        self.action_mapping = [[1, 0], [0, 1], [-1, 0], [0, -1]]

        self.tmaze_map = np.zeros(
            (3 + 2, self.oracle_length + self.corridor_length + 1 + 2), dtype=bool
        )
        self.bias_x, self.bias_y = 1, 2
        self.tmaze_map[self.bias_y, self.bias_x : -self.bias_x] = True  # corridor
        self.tmaze_map[
            [self.bias_y - 1, self.bias_y + 1], -self.bias_x - 1
        ] = True  # goal candidates
        # print(self.tmaze_map.astype(np.int32))

        obs_dim = 2 if self.ambiguous_position else 3
        if self.expose_goal:  # test Markov policies
            assert self.ambiguous_position is False
        if self.add_timestep:
            obs_dim += 1

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self.deterministic = deterministic
        if deterministic:
            self.goal_y = np.random.choice([-1, 1])

    def position_encoding(self, x: int, y: int, goal_y: int):
        if x == 0:
            # oracle position
            if not self.oracle_visited:
                # only appear at first
                exposure = goal_y
                self.oracle_visited = True
            else:
                exposure = 0

        if self.ambiguous_position:
            if x == 0:
                # oracle position
                return [0, exposure]
            elif x < self.oracle_length + self.corridor_length:
                # intermediate positions (on the corridor)
                return [0, 0]
            else:
                # T-junction or goal candidates
                return [1, y]
        else:
            if self.expose_goal:
                return [x, y, goal_y if self.oracle_visited else 0]
            if x == 0:
                # oracle position
                return [x, y, exposure]
            else:
                return [x, y, 0]

    def timestep_encoding(self):
        return (
            [
                self.time_step,
            ]
            if self.add_timestep
            else []
        )

    def get_obs(self):
        return np.array(
            self.position_encoding(self.x, self.y, self.goal_y)
            + self.timestep_encoding(),
            dtype=np.float32,
        )

    def reward_fn(self, done: bool, x: int, y: int, goal_y: int):
        if done:  # only give bonus at the final time step
            return float(y == goal_y) * self.goal_reward
        else:
            # a penalty (when t > o) if x < t - o (desired: x = t - o)
            rew = float(x < self.time_step - self.oracle_length) * self.penalty
            if x == 0:
                return rew + self.distract_reward
            else:
                return rew

    def step(self, action):
        self.time_step += 1
        assert self.action_space.contains(action)

        # transition
        move_x, move_y = self.action_mapping[action]
        if self.tmaze_map[self.bias_y + self.y + move_y, self.bias_x + self.x + move_x]:
            # valid move
            self.x, self.y = self.x + move_x, self.y + move_y

        done = self.time_step >= self.episode_length
        rew = self.reward_fn(done, self.x, self.y, self.goal_y)
        return self.get_obs(), rew, done, {}

    def reset(self):
        self.x, self.y = self.oracle_length, 0
        if not self.deterministic:
            self.goal_y = np.random.choice([-1, 1])

        self.oracle_visited = False
        self.time_step = 0
        return self.get_obs()


class TMazeClassicPassive(TMazeBase):
    def __init__(
        self,
        corridor_length: int = 10,
        goal_reward: float = 1.0,
        penalty: float = 0.0,
        distract_reward: float = 0.0,
    ):
        """
        Classic TMaze with Passive Memory
            assert episode_length == corridor_length + 1
            (Bakker, 2001): ambiguous_position = True. penalty = 0
                This is too hard even for T = 10 for vanilla agents because the exploration is extremely hard.
                This tests both memory and exploration
            **(tmaze_classic; this work)**: based on (Bakker, 2001), set penalty < 0
                Unit-tests memory
        """
        super().__init__(
            episode_length=corridor_length + 1,
            corridor_length=corridor_length,
            goal_reward=goal_reward,
            penalty=penalty,
            distract_reward=distract_reward,
            expose_goal=False,
            ambiguous_position=True,
            add_timestep=False,
        )


class TMazeClassicActive(TMazeBase):
    def __init__(
        self,
        corridor_length: int = 10,
        goal_reward: float = 1.0,
        penalty: float = 0.0,
        distract_reward: float = 0.0,
    ):
        """
        Classic TMaze with Active Memory
            assert episode_length == corridor_length + 1 + 2o
            where o is the length between the starting point and oracle that gives the goal information
            TMazeClassicPassive is a special case of o = 0.
        """
        oracle_length = 1
        super().__init__(
            episode_length=corridor_length + 2 * oracle_length + 1,
            corridor_length=corridor_length,
            oracle_length=oracle_length,
            goal_reward=goal_reward,
            penalty=penalty,
            distract_reward=distract_reward,
            expose_goal=False,
            ambiguous_position=True,
            add_timestep=False,
        )


class TMazeProblem(Problem):
    def __init__(self, 
                 maze_type,
                 max_corridor_length,
                 min_corridor_length=10,
                 penalty: float = -0.1,
                 **kwargs):
        assert min_corridor_length < max_corridor_length
        self.max_corridor_length = max_corridor_length
        self.min_corridor_length = min_corridor_length
        self.penalty = penalty
        self.maze_type = maze_type
        self.min_val, self.max_val = -1, 1
        self.current_obs = None
        self.optimal_action = None
        self.current_reward = 0.0
        
        if self.maze_type == "active":
            max_episode_length = max_corridor_length + 2 * 1 + 1  # +2*oracle_length+1
        else:  # passive
            max_episode_length = max_corridor_length + 1
        kwargs.setdefault("seq_len", max_episode_length) # override this for setting max episode length

        super().__init__(
            observation_shape=(2,),  # ambiguous position gives 2D obs
            observation_type=float,
            **kwargs
        )
        
        self.reset()
        self.action_mapping = self.samples[0].action_mapping
    
    def _create_env(self, corridor_length):
        if self.maze_type == "active":
            return TMazeClassicActive(
                corridor_length=corridor_length,
                penalty=self.penalty
            )
        else:  # passive
            return TMazeClassicPassive(
                corridor_length=corridor_length,
                penalty=self.penalty
            )
    
    def _get_optimal_action(self, env, obs):
        if self.maze_type == "active":
            # Active TMaze optimal policy based on observation
            if obs[0] == 0 and obs[1] == 0:
                # At starting position or corridor - go right (or left to oracle on first step)
                if env.time_step == 0:
                    return 2  # left to oracle
                else:
                    return 0  # right through corridor
            elif obs[0] == 0 and obs[1] != 0:
                # At oracle position with goal info - go right
                return 0  # right
            elif obs[0] == 1 and obs[1] == 0:
                # At T-junction - choose based on remembered goal
                if hasattr(env, 'goal_y'):
                    if env.goal_y == -1:
                        return 3  # down
                    else:
                        return 1  # up
                else:
                    return 1  # default up
            else:
                # At goal position or other - stay
                return 0  # arbitrary
        else:
            # Passive TMaze optimal policy
            if obs[0] == 1 and obs[1] == 0:
                # At T-junction - choose based on goal info from start
                if hasattr(env, 'goal_y'):
                    if env.goal_y == -1:
                        return 3  # down
                    else:
                        return 1  # up
                else:
                    return 1  # default up
            else:
                return 0  # right through corridor
    
    def get_sample(self, idx):
        env = self.samples[self.sidx]
        if idx == 0:
            self.current_obs = env.reset()
            self.current_reward = 0.0
        else:
            self.current_obs, self.current_reward, _, _ = env.step(self.optimal_action)

        self.optimal_action = self._get_optimal_action(env, self.current_obs)
        return self.current_obs

    def get_reward(self):
        return float(self.current_reward)

    def get_target(self, t):
        if self.current_obs is None:
            return np.array([0], dtype=np.float32)  # default action

        return np.array(self.action_mapping[self.optimal_action], dtype=np.float32)
    
    def reset(self):
        if self.samples is None:
            self.samples = []
            for _ in range(self.num_variations):
                corridor_length = random.randint(self.min_corridor_length, 
                                                 self.max_corridor_length)
                env = self._create_env(corridor_length)
                self.samples.append(env)
            self.sidx = 0
        else:
            if self.randomize_reset:
                self.sidx = random.randint(0, self.num_variations - 1)
            else:
                self.sidx = (self.sidx + 1) % self.num_variations
        
        # Update seq_len dynamically based on current variation's episode length
        env = self.samples[self.sidx]
        self.seq_len = env.episode_length
    
    def reset_sidx(self, sidx):
        self.sidx = sidx
        # Update seq_len dynamically based on current variation's episode length
        env = self.samples[self.sidx]
        self.seq_len = env.episode_length


class TMazeEvaluator(Evaluator):
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
        oracle_priorities = []
        corridor_priorities = []
        junction_priorities = []
        mse_episode = []
        for _ in range(self.evaluation_rollouts):
            process.reset()
            mse_his = []
            for idx in range(process.problem.seq_len):
                smp, t = process.get_obs(idx)
                target = process.get_target(t)
                p, _ = selector.get_priority(smp)
                process.set_action(smp, p)
                if smp[0] == 0 and smp[1] != 0:  # Oracle position
                    oracle_priorities.append(p)
                elif smp[0] == 0 and smp[1] == 0:  # Corridor position
                    corridor_priorities.append(p)
                elif smp[0] == 1:  # Junction or goal position
                    junction_priorities.append(p)
                
                pred_target = proxy.get_action(process.get_buffer()
                                                .unsqueeze(0).to(proxy.device)).cpu().squeeze()
                mse_his.append(((target - pred_target) ** 2).mean())
            
            avg_mse = sum(mse_his) / len(mse_his)
            mse_episode.append(avg_mse)
        max_mse = max(mse_episode)
        print(f"terminal - buffer state: {process.get_buffer()} | target: {target} | predicted: {pred_target}")
        print(f"avg oracle priority: {sum(oracle_priorities)/len(oracle_priorities)}")
        print(f"avg corridor priority: {sum(corridor_priorities)/len(corridor_priorities)}")
        print(f"avg junction priority: {sum(junction_priorities)/len(junction_priorities)}")
        print(f"max mse: {max_mse}")

        return max_mse


class TMazePolicyEvaluator(Evaluator):
    """Policy-stage evaluator: selector+policy eval via the rollout registry in an
    isolated subprocess; early-stops on success rate."""

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
        sr = result
        print(f"Policy Eval — SR: {sr:.3f}")
        if self.logging:
            wandb.log({"eval/policy_sr": sr, "eval_step": self.step})
            self.step += 1
        return sr > self.sr_threshold
