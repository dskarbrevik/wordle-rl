from wordle_rl import WordleEnv
import gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3 import HerReplayBuffer, DQN
import sys
import torch
import os
import time

from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3 import HerReplayBuffer, DQN

def main(train_steps):
    start = time.time()
    env = WordleEnv("config.json")
    if os.path.exists("./wordle_rl_model_dqn.zip"):
        model = DQN.load("./wordle_rl_model_dqn", 
                        env,
                        gamma=1,
                        replay_buffer_class=HerReplayBuffer,
                        # Parameters for HER
                        replay_buffer_kwargs=dict(
                            n_sampled_goal=1000,
                            goal_selection_strategy='future',
                            online_sampling=True,
                            max_episode_length=6),
                         verbose=1, 
                         device=torch.device(0))
        print("Using pre-trained model as starting point.")
    else:
        model = DQN(policy='MultiInputPolicy',
                    gamma=1,
                    replay_buffer_class=HerReplayBuffer,
                    # Parameters for HER
                    replay_buffer_kwargs=dict(
                    n_sampled_goal=1000,
                    goal_selection_strategy='future',
                    online_sampling=True,
                    max_episode_length=6),
                    env=env, 
                    verbose=1,
                    device=torch.device(0))
        print("No existing model found, starting from scratch.")

    model.learn(total_timesteps=int(train_steps))

    model.save("wordle_rl_model_ dqn")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)

    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    print(f"total time to train {int(train_steps)} timesteps = {round((time.time()-start)/(60*60),2)} hours.")

if __name__=='__main__':
    main(sys.argv[-1])