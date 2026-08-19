from stable_baselines3 import PPO
import gymnasium as gym

if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    print("Training Standard Baseline PPO on default parameters...")
    baseline_model = PPO("MlpPolicy", env, verbose=1)
    baseline_model.learn(total_timesteps=50_000)

    save_location = "checkpoints/naive_cartpole_ppo"
    baseline_model.save(save_location)
    print(f"Training complete! Saved as '{save_location}.zip'.")
