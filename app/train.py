from wordle_rl import WordleEnv
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
import sys
import torch
import os

def main(train_steps):
    env = WordleEnv("config.json")
    if os.path.exists("./wordle_rl_model.zip"):
        model = A2C.load("wordle_rl_model.zip", env, verbose=1, device=torch.device(0))
        print("Using pre-trained model as starting point.")
    else:
        model = A2C('MlpPolicy', env, verbose=1, device=torch.device(0))
        print("No existing model found, starting from scratch.")

    model.learn(total_timesteps=int(train_steps))

    model.save("wordle_rl_model")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)

    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

if __name__=='__main__':
    main(sys.argv[-1])