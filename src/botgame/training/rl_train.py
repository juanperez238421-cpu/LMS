import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from botgame.training.rl_env import BotGameEnv

def train_rl_agent(log_dir: str = "logs/ppo", model_save_path: str = "artifacts/ppo_policy.zip", total_timesteps: int = 100000) -> None:
    """
    Trains an RL agent using PPO with Stable-Baselines3.
    Args:
        log_dir: Directory to save training logs.
        model_save_path: Path to save the trained PPO model.
        total_timesteps: Total number of timesteps to train the agent.
    """
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Create the environment
    # Use make_vec_env for vectorized environments, which is common for SB3
    # For a single bot, we can just pass the BotGameEnv directly
    env = BotGameEnv(bot_id="rl_bot_0")
    # env = make_vec_env(lambda: BotGameEnv(bot_id="rl_bot_0"), n_envs=1)

    # Initialize the PPO agent
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)

    # Define callbacks for evaluation and early stopping
    # eval_env = make_vec_env(lambda: BotGameEnv(bot_id="rl_bot_eval"), n_envs=1)
    eval_env = BotGameEnv(bot_id="rl_bot_eval")

    # Stop training if the mean reward over 5 episodes is >= 10 (example threshold)
    callback_on_best_model = StopTrainingOnRewardThreshold(reward_threshold=10, verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=os.path.join(log_dir, "best_model"),
                                 log_path=log_dir,
                                 eval_freq=5000,
                                 deterministic=True,
                                 render=False,
                                 callback_on_new_best=callback_on_best_model)

    print(f"Starting RL training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Save the final model
    model.save(model_save_path)
    print(f"Trained PPO model saved to {model_save_path}")

def main():
    train_rl_agent()

if __name__ == "__main__":
    main()
