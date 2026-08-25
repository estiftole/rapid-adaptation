import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

if __name__ == "__main__":
    env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave', 'kettle'], render_mode="human")
    _, info = env.reset()

    for _ in range(100):
        action = env.action_space.sample()
        env.step(action)

    env.close()
