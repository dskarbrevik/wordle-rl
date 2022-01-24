import gym
from gym import spaces
import json
import logging
import numpy as np

LOG_LEVEL_DICT = {
    "DEBUG": logging.DEBUG, # note: DEBUG will catch nx.graph errors & can be quite verbose
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR
}

class WordleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config_path):
        super(WordleEnv, self).__init__() # inherit gym env class

        # load a config file
        with open(config_path, 'r') as file:
            self.config = json.load(file)
            
        # logging!
        logging.basicConfig(
            format="%(levelname)s - %(message)s",
            level=LOG_LEVEL_DICT.get(self.config['rl_env']['VERBOSITY'])
            )

        # get our word lists for the game
        self.valid_words = self._get_words(self.config['word_file_paths']['valid_words_file_path'])
        self.game_words = self._get_words(self.config['word_file_paths']['valid_words_file_path'])

        # defining some of the game parameters of Wordle
        self.obs_height = 6
        self.obs_width = 5
        self.obs_channels = 2 # one for word guesses and another for wordle clues
        self.num_actions = len(self.valid_words)

        # for tracking the state of the game
        self.win_reward = int(self.config['rl_env']['WIN_REWARD'])
        self.right_letter_reward = int(self.config['rl_env']['RIGHT_LETTER_REWARD'])
        self.right_position_reward = int(self.config['rl_env']['RIGHT_POSITION_REWARD'])
        self.penalty = int(self.config['rl_env']['PENALTY'])
        self.obs_state = np.zeros(shape=(self.obs_channels,self.obs_height,self.obs_width))
        self.current_step = 0
        self.total_reward = 0
        self.action_word = ""
        self.used_words = []
        self.current_word = str(np.random.choice(self.game_words))
        self.current_word_vec = [ord(char) - 96 for char in self.current_word]

        # gym needs you to define the action space and observation space
        # define action space
        self.action_space = spaces.Discrete(self.num_actions)

        # define observation space
        # 0-25 for letters of alphabet
        self.observation_space = spaces.Box(low=0, high=29, shape=
                        (self.obs_channels, self.obs_height, self.obs_width), dtype=np.uint8)

    # @property
    # def action_space(self):
    #     for word in self.used_words:
    #         if word in 
    #     return


    def step(self, action):
    # Execute one time step within the environment
        self._take_action(action)
        reward = self._calc_reward()
        self.total_reward += reward
        self.current_step += 1
        done = self._check_done() # calculation of game state
        return self.obs_state, reward, done, {}


    def reset(self):
    # Reset the state of the environment to an initial state
        self.obs_state = np.zeros(shape=(self.obs_channels,self.obs_height,self.obs_width))
        self.current_step = 0
        self.total_reward = 0
        self.current_word = str(np.random.choice(self.game_words))
        self.current_word_vec = [ord(char) - 96 for char in self.current_word]
        return self.obs_state


    def render(self, mode='human', close=False):
    # Render the environment to the screen
        print(f"Correct word: {self.current_word}")
        print(f"Guessed word: {self.action_word}")
        print(f"Total reward: {self.total_reward}")
        if self._check_done():
            print(self.obs_state)


    def _take_action(self,action):
        action_word = self.valid_words[action]
        self.action_word = action_word
        action_vec = [ord(char) - 96 for char in action_word]
        result_vec = []
        for i,num in enumerate(action_vec):
            if action_vec[i]==self.current_word_vec[i]:
                result_vec.append(29) # got the letter and position right!
            elif num in self.current_word_vec: 
                result_vec.append(28) # got the letter right
            else:
                result_vec.append(27) # letter is not in word
        
        self.obs_state[0,self.current_step,:] = action_vec
        self.obs_state[1,self.current_step,:] = result_vec


    def _calc_reward(self):
        reward = 0
        win = True
        result = self.obs_state[1,self.current_step,:]

        if self.current_step > 0:
            if np.array_equal(result,self.obs_state[1,self.current_step-1,:]):
                return(self.penalty)
        for num in result:
            if num == 28:
                win = False
                reward += self.right_letter_reward
            elif num == 29:
                reward += self.right_position_reward
            else:
                win = False
        if win:
            reward = round(self.win_reward*(1/(self.current_step+1)))
        return reward


    def _check_done(self):
        if self.current_step==6:
            return True
        for vec in self.obs_state[1,:,:]:
            for num in vec:
                if num!=29:
                    break
            else:
                return True
        return False


    def _get_words(self,word_file_path):
        with open(word_file_path,"r") as file:
            word_list = file.read().splitlines()
        return word_list
