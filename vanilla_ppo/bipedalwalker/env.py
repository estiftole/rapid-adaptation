import gymnasium as gym
from gymnasium.envs.box2d.bipedal_walker import BipedalWalker, LEG_H
import numpy as np

class CustomBipedalWalkerEnv(BipedalWalker):
    def __init__(self, render_mode=None, randomize_freq=1):
        super().__init__(render_mode=render_mode)

        self.friction_range = (0.5, 2.5)
        self.density_range = (2.0, 7.0)
        self.leg_scale_range = (0.7, 1.3)

        self.randomize_freq = randomize_freq
        self.episode_counter = 0

        self.true_friction = 2.5
        self.true_density = 5.0
        self.true_leg_scale = 1.0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        if self.episode_counter % self.randomize_freq == 0:
            self.true_friction = np.random.uniform(*self.friction_range)
            self.true_density = np.random.uniform(*self.density_range)
            self.true_leg_scale = np.random.uniform(*self.leg_scale_range)

        self.episode_counter += 1

        custom_leg_h = (34.0 / 30.0) * self.true_leg_scale
        import gymnasium.envs.box2d.bipedal_walker as bw
        bw.LEG_H = custom_leg_h

        obs, info = super().reset(seed=seed, options=options)


        for fixture in self.hull.fixtures:
            fixture.density = float(self.true_density)
        self.hull.ResetMassData()

        for poly in self.terrain:
            poly.fixtures[0].friction = float(self.true_friction)


        info["true_friction"] = self.true_friction
        info["true_density"] = self.true_density
        info["true_leg_scale"] = self.true_leg_scale

        return obs, info

if __name__ == "__main__":
    env = CustomBipedalWalkerEnv()
    obs, info = env.reset()

    print("Environment successfully initialized!")
    print(f"Observation dimension: {obs.shape}")
    print(f"Privileged Info -> Friction: {info['true_friction']:.2f}, Density: {info['true_density']:.2f}, Leg scale: {info['true_leg_scale']:.2f}")
    env.close()
