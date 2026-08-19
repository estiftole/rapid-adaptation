import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from env import CustomCartPoleEnv

class PrivilegedObservationWrapper(gym.ObservationWrapper):
    """
    Appends the true mass and length from the info dict onto the
    end of the observation vector, making it a 6-dim vector for the Model.
    """
    def __init__(self, env):
        super().__init__(env)
        # Original observation space is 4D. We expand it by 2 for mass & length.
        low = np.append(env.observation_space.low, [0.05, 0.2]).astype(np.float32)
        high = np.append(env.observation_space.high, [0.5, 1.5]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        mass = self.env.unwrapped.masspole
        length = self.env.unwrapped.length
        return np.append(observation, [mass, length]).astype(np.float32)

if __name__ == "__main__":
    env = CustomCartPoleEnv()
    env = PrivilegedObservationWrapper(env)

    print("Training Universal Policy with PPO...")
    model = PPO("MlpPolicy", env, verbose=1)
    _ = model.learn(total_timesteps=50_000)

    save_location = "checkpoints/univ_cartpole_ppo"
    model.save(save_location)
    print(f"Model training complete! Model saved as '{save_location}.zip'.")
