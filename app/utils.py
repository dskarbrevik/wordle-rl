def play_a_tfagents_game(env,agent):
    for i in range(6):
        if i == 0:
            time_step = env.reset()
            print(f"This game's correct word is: '{env.pyenv.envs[0].gym.current_word}'\n")
            print("RL Agent's guesses:")
    
        action_step = agent.policy.action(time_step)
        chosen_word = env.pyenv.envs[0].gym.valid_words[int(action_step.action.numpy())]
        time_step = env.step(action_step.action)
        for letter in chosen_word:
            print(f"{letter} ",end='')
        print(" "*5,end='')
        print(f"Reward = {int(time_step.reward.numpy())}")
        print("- "*5)

    print("\nFinal Result:")
    print(time_step[3].numpy().reshape((2,6,5)))
    print(env)



def play_a_stablebaselines_game(env,model):
    #TODO
    pass