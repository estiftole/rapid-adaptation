import gymnasium as gym
from stable_baselines3 import PPO

if __name__ == "__main__":
    env = gym.make("BipedalWalker-v3")

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],
            vf=[256, 256]
        )
    )

    train_steps = 10_000_000
    print("Training Naive Baseline Policy on standard BipedalWalker-v3...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        ent_coef=0.005,
        gae_lambda=0.95,
        gamma=0.99,
        verbose=1
    )

    model.learn(total_timesteps=train_steps)
    save_location = "checkpoints/naive_bipedalwalker_ppo"
    model.save(save_location)
    print(f"Training complete! Saved as '{save_location}.zip'.")
