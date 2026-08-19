import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
import numpy as np

class CustomCartPoleEnv(CartPoleEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self, seed=None, options: None =None, param_seed:int=42):
        super().reset(seed=seed)

        # randomize pole mass and length for this episode
        self.set_env_params(*randomize_env_params())

        # standard CartPole reset state
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None

        observation = np.array(self.state, dtype=np.float32)
        info = {
            "true_mass": self.masspole,
            "true_length": self.length
        }
        return observation, info

    def set_env_params(self, masspole: float, length: float):
        self.masspole, self.length = masspole, length
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length

def randomize_env_params(seed: int=None):
    mass_range = (0.05, 0.5)
    length_range = (0.2, 1.5)
    if seed:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    return rng.uniform(low=mass_range[0], high=mass_range[1]), rng.uniform(low=length_range[0], high=length_range[1])


if __name__ == "__main__":
    gym.register(
        id="CustomCartPole-v1",
        entry_point=CustomCartPoleEnv,
        max_episode_steps=500,
    )

    masspole, length = randomize_env_params()
    env = gym.make(
        "CustomCartPole-v1",
        render_mode="human")

    obs, info = env.reset(seed=42)
    print("Initialized Environment successfully!")
    print(f"Initial Observation: {obs}")
    print(f"Privileged Info -> Mass: {info['true_mass']:.4f}, Length: {info['true_length']:.4f}")
    env.close()
