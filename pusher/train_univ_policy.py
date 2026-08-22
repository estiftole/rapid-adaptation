import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from env import CustomPusherEnv
import os

class PrivilegedObservationWrapper(gym.ObservationWrapper):
    """
    Appends the true mass and length from the info dict onto the
    end of the observation vector, making it a 6-dim vector for the Model.
    """
    def __init__(self, env):
        super().__init__(env)

        low = np.append(env.observation_space.low, [0.2, 0.03, 0.3]).astype(np.float32)
        high = np.append(env.observation_space.high, [5., 0.08, 1.7]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        putt_mass = self.env.unwrapped.putt_mass
        putt_size = self.env.unwrapped.putt_size
        forearm_scale = self.env.unwrapped.forearm_scale
        return np.append(obs, [putt_mass, putt_size, forearm_scale]).astype(np.float32)


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train_steps = 1_000_000
    env = CustomPusherEnv(randomize_freq=10)
    env = PrivilegedObservationWrapper(env)

    policy_kwargs = dict(net_arch=dict(pi=[256, 256],vf=[256, 256]))

    print(f"Training Universal Policy on standard Pusher-v5 ({train_steps} steps)...")
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1
    )

    model.learn(total_timesteps=train_steps)

    save_path = "checkpoints/univ_pusher_ppo"
    model.save(save_path)
    print(f"\nUniversal Baseline training complete! Model saved to '{save_path}.zip'.")

    env.close()
