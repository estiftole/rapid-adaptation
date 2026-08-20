import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import CustomBipedalWalkerEnv

class PrivilegedObservationWrapper(gym.ObservationWrapper):
    """
    Appends true_friction (0.1 to 3.0) and true_density (1.0 to 10.0)
    onto the 24-dim BipedalWalker observation vector -> 26-dim vector.
    """
    def __init__(self, env):
        super().__init__(env)

        low = np.append(env.observation_space.low, [-1.0, -1.0, -1.0]).astype(np.float32)
        high = np.append(env.observation_space.high, [1.0, 1.0, 1.0]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        f_norm = 2.0 * (self.env.unwrapped.true_friction - 0.5) / (2.5 - 0.5) - 1.0
        d_norm = 2.0 * (self.env.unwrapped.true_density - 2.0) / (7.0 - 2.0) - 1.0
        l_norm = 2.0 * (self.env.unwrapped.true_leg_scale - 0.7) / (1.3 - 0.7) - 1.0
        return np.append(obs, [f_norm, d_norm, l_norm]).astype(np.float32)

def make_env():
    env = CustomBipedalWalkerEnv(randomize_freq=10)
    env = PrivilegedObservationWrapper(env)
    return env

if __name__ == "__main__":
    # Wrap in DummyVecEnv and VecNormalize (CRITICAL for continuous locomotion)
    train_steps = 1_000_000
    vec_env = DummyVecEnv([make_env])
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    # 2. Configure PPO for continuous locomotion (256x256 MLP)
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],
            vf=[256, 256]
        )
    )

    print(f"Training Universal Bipedal Teacher Policy ({train_steps} steps)...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2e-4,        # Slightly lower learning rate for stability
        n_steps=4096,              # Doubled rollout buffer (more trajectory context)
        batch_size=128,            # Larger mini-batches to handle trajectory variance
        n_epochs=10,
        ent_coef=0.005,            # FORCES exploration; stops premature "ducking"
        gae_lambda=0.95,
        gamma=0.99,
        verbose=1
    )

    # 3. Learn policy
    model.learn(total_timesteps=train_steps)
    save_location = "checkpoints/univ_bipedalwalker_ppo"
    model.save(save_location)
    vec_env = model.get_vec_normalize_env()
    vec_env.save("checkpoints/univ_bipedal_vecnormalize.pkl")
    print(f"Model training complete! Model saved as '{save_location}.zip'.")
