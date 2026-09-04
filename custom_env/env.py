import gymnasium as gym

class CustomEvnrionment(gym.Env):
    def __init__(self): pass
    def _get_obs_(self): pass
    def _get_reward_(self): pass
    def reset(self, seed, options): pass
    def step(self, action): pass
    def render(self): pass
    def close(self): pass

# uv run -m mujoco.viewer --mjcf=custom_env/custom_models/model.xml
