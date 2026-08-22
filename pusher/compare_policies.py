import numpy as np
from stable_baselines3 import PPO
from env import CustomPusherEnv
from train_univ_policy import PrivilegedObservationWrapper
import gymnasium as gym
import mujoco

def evaluate_model(model_type: str, num_episodes: int =5, max_steps: int =200, render: int =False) -> list:
    print(f"Evaluating {model_type} Model")

    # env = gym.make("Pusher-v5", render_mode=("human" if render else None))
    env = CustomPusherEnv(render_mode=("human" if render else None))
    env = PrivilegedObservationWrapper(env)

    if model_type == "UNIVERSAL":
        model = PPO.load("checkpoints/univ_pusher_ppo")
    else:
        model = PPO.load("checkpoints/naive_pusher_ppo")

    results = []

    for ep in range(num_episodes):
        obs, start_info = env.reset()
        print(f"Episode params: {start_info}")
        done = False
        steps = 0

        while not done:
            # action, _ = model.predict(obs, deterministic=True)
            if model_type == "UNIVERSAL":
                action, _ = model.predict(obs, deterministic=True)
            else:
                action, _ = model.predict(obs[:-3], deterministic=True)

            obs, reward, done, _, info = env.step(action)
            steps += 1
            if steps >= max_steps: break

            # if render:
                # time.sleep(0.01)

        results.append({
            # "mass": start_info['true_mass'],
            # "length": start_info['true_length'],
            "steps": steps
        })

    env.close()
    return results

if __name__ == "__main__":
    max_steps = 500
    # num_episodes = 200
    # render = False
    num_episodes = 5
    render = True
    print(f"Running stress test comparison ({num_episodes} episodes each)...")

    naive_results = evaluate_model("NAIVE", num_episodes, max_steps, render)
    naive_avg = np.mean([r['steps'] for r in naive_results])
    print(f"\n--- Results (Average Steps Survived) ---")
    print(f"Naive Policy:     {naive_avg:.2f} steps")
    naive_fails = [r for r in naive_results if r['steps'] < max_steps]
    print(f"\nNaive Policy failed on {len(naive_fails)} out of {num_episodes} episodes.")


    # univ_results = evaluate_model("UNIVERSAL", num_episodes, max_steps, render)
    # univ_avg = np.mean([r['steps'] for r in univ_results])
    # print(f"\n--- Results (Average Steps Survived) ---")
    # print(f"Universal Policy: {univ_avg:.2f} steps")
    # univ_fails = [r for r in univ_results if r['steps'] < max_steps]
    # print(f"Universal Policy failed on {len(univ_fails)} out of {num_episodes} episodes.")
