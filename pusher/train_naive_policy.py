import os
import gymnasium as gym
from stable_baselines3 import PPO

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train_steps = 1_000_000
    env = gym.make("Pusher-v5")

    policy_kwargs = dict(net_arch=dict(pi=[256, 256],vf=[256, 256]))

    print(f"Training Naive Baseline Policy on standard Pusher-v5 ({train_steps} steps)...")
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1
    )

    model.learn(total_timesteps=train_steps)

    save_path = "checkpoints/naive_pusher_ppo"
    model.save(save_path)
    print(f"\nNaive Baseline training complete! Model saved to '{save_path}.zip'.")

    env.close()
