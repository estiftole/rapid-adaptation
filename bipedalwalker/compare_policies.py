import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import CustomBipedalWalkerEnv
from train_univ_policy import PrivilegedObservationWrapper
import gymnasium as gym

def evaluate_model(model_type: str, num_episodes: int = 5, max_steps: int = 500, render: bool = False) -> list:
    print(f"\nEvaluating {model_type} Model...")

    # 1. Create the base environment inside a callable for DummyVecEnv
    def make_env():
        # env = gym.make("BipedalWalker-v3", render_mode=("human" if render else None))
        env = CustomBipedalWalkerEnv(render_mode=("human" if render else None))
        env = PrivilegedObservationWrapper(env)
        return env

    # Wrap in DummyVecEnv (SB3 standard for vectorization)
    vec_env = DummyVecEnv([make_env])

    # 2. Apply VecNormalize ONLY to Universal, load stats, and freeze them
    if model_type == "UNIVERSAL":
        model = PPO.load("checkpoints/univ_bipedalwalker_ppo")
        # Load the normalization stats saved during training
        vec_env = VecNormalize.load("checkpoints/univ_bipedal_vecnormalize.pkl", vec_env)
        vec_env.training = False       # Freeze running mean/std
        vec_env.norm_reward = False    # Disable reward normalization so we can read real scores
    else:
        model = PPO.load("checkpoints/naive_bipedalwalker_ppo")
        # The Naive model uses vec_env as-is, without VecNormalize

    results = []

    for ep in range(num_episodes):
        obs = vec_env.reset()

        # Extract privileged parameters directly from the unwrapped environment
        raw_env = vec_env.envs[0].unwrapped
        # print(f"Episode {ep+1} params -> Friction: {raw_env.true_friction:.2f} Density: {raw_env.true_density:.2f}, Leg Scale: {raw_env.true_leg_scale:.2f}")

        done = False
        steps = 0

        while not done:
            # 3. Handle state dimensions. obs shape is now (1, 27) due to DummyVecEnv
            if model_type == "UNIVERSAL":
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Slice the batch dimension and feature dimension for the Naive model
                action, _ = model.predict(obs[:, :24], deterministic=True)

            # 4. VecEnv step returns 4 values; dones is a boolean array
            obs, rewards, dones, infos = vec_env.step(action)
            steps += 1

            # Extract the boolean done flag for the single environment
            done = dones[0] or (steps >= max_steps)

            if render:
                time.sleep(0.01)

        results.append({
            # "friction": raw_env.true_friction,
            # "density": raw_env.true_density,
            # "leg_scale": raw_env.true_leg_scale,
            "steps": steps
        })

    vec_env.close()
    return results

if __name__ == "__main__":
    max_steps = 500
    num_episodes = 5
    render = True
    # num_episodes = 50
    # render = False

    print(f"Running stress test comparison ({num_episodes} episodes each)...")

    naive_results = evaluate_model("NAIVE", num_episodes, max_steps, render)
    naive_avg = np.mean([r['steps'] for r in naive_results])
    print(f"\n--- Results (Average Steps Survived) ---")
    print(f"Naive Policy:     {naive_avg:.2f} steps")
    naive_fails = [r for r in naive_results if r['steps'] < max_steps]
    print(f"\nNaive Policy failed on {len(naive_fails)} out of {num_episodes} episodes.")


    univ_results = evaluate_model("UNIVERSAL", num_episodes, max_steps, render)
    univ_avg = np.mean([r['steps'] for r in univ_results])
    print(f"\n--- Results (Average Steps Survived) ---")
    print(f"Universal Policy: {univ_avg:.2f} steps")
    univ_fails = [r for r in univ_results if r['steps'] < max_steps]
    print(f"Universal Policy failed on {len(univ_fails)} out of {num_episodes} episodes.")
