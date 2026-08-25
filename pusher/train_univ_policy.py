import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from env import CustomPusherEnv
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
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

        # low = np.append(env.observation_space.low, [-1.0, -1.0, -1.0]).astype(np.float32)
        # high = np.append(env.observation_space.high, [1.0, 1.0, 1.0]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        putt_mass = self.env.unwrapped.putt_mass
        putt_size = self.env.unwrapped.putt_size
        forearm_scale = self.env.unwrapped.forearm_scale
        return np.append(obs, [putt_mass, putt_size, forearm_scale]).astype(np.float32)

        # putt_mass_norm = 2.0 * (self.env.unwrapped.putt_mass - 0.2) / (5. - 0.2) - 1.0
        # putt_size_norm = 2.0 * (self.env.unwrapped.putt_size - 0.03) / (0.08 - 0.03) - 1.0
        # forearm_scale_norm = 2.0 * (self.env.unwrapped.forearm_scale - 0.3) / (1.7 - 0.3) - 1.0

        # return np.append(obs, [putt_mass_norm, putt_size_norm, forearm_scale_norm]).astype(np.float32)



def make_env():
    env = CustomPusherEnv(randomize_freq=15)
    env = PrivilegedObservationWrapper(env)
    return env

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train_steps = 3_000_000
    # vec_env = DummyVecEnv([make_env])
    # env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    env = make_env()

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],       # 2-layer policy network
            vf=[512, 512]        # Larger 2-layer value network for dynamic physics
        ),
        log_std_init=-1.0        # Keeps initial joint control tame
    )


    print(f"Training Universal Policy on standard Pusher-v5 ({train_steps} steps)...")
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1.5e-4,    # Reduced learning rate for larger network stability
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1
    )

    model.learn(total_timesteps=train_steps)

    save_path = "checkpoints/univ_pusher_ppo"
    model.save(save_path)
    # vec_env = model.get_vec_normalize_env()
    # vec_env.save("checkpoints/univ_pusher_vecnormalize.pkl")
    print(f"\nUniversal Baseline training complete! Model saved to '{save_path}.zip'.")
    env.close()
