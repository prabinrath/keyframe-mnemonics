from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import wandb


class WandbCallback(BaseCallback):
    """Custom callback for logging PPO metrics to wandb after each iteration."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        if wandb.run is None:
            raise RuntimeError("Initialize logging: wandb.init() must be called before using WandbCallback")
        wandb.define_metric("ppo_step")
        wandb.define_metric("ppo/*", step_metric="ppo_step")
        self.step = 1
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        """Log metrics at the end of each PPO iteration."""
        # Get episode statistics from ep_info_buffer
        if len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
            ep_lengths = [ep_info['l'] for ep_info in self.model.ep_info_buffer]
            wandb.log({
                'ppo/mean_episode_reward': np.mean(ep_rewards),
                'ppo/mean_episode_length': np.mean(ep_lengths),
                'ppo_step': self.step
            })
            self.step += 1
