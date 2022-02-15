import gym
from gym import spaces
import json
import logging
import numpy as np
import random

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
        self.simple_game = self.config['rl_env']['SIMPLIFY_GAME']
        if self.simple_game:
            self.valid_words = self._get_words(self.config['word_file_paths']['valid_words_file_path'],simplify=True)
            self.game_words = self.valid_words
        else:
            self.valid_words = self._get_words(self.config['word_file_paths']['valid_words_file_path'])
            self.game_words = self._get_words(self.config['word_file_paths']['valid_words_file_path'])

        # defining some of the game parameters of Wordle
        self.obs_height = 6
        self.obs_width = 5
        self.obs_channels = 2 # one for word guesses and another for wordle clues
        self.num_actions = len(self.valid_words)
        self.tf_agents = self.config['rl_env']['TF_AGENTS']
        self.goal_env = self.config['rl_env']['GOAL_ENV']

        # for tracking the state of the game
        self.win_reward = int(self.config['rl_env']['WIN_REWARD'])
        self.right_letter_reward = int(self.config['rl_env']['RIGHT_LETTER_REWARD'])
        self.right_position_reward = int(self.config['rl_env']['RIGHT_POSITION_REWARD'])
        self.penalty = int(self.config['rl_env']['PENALTY'])


        ## THREE DIFFERENT SCENARIOS ARE COVERED HERE... should have split it into three files

        # used with TF Agents environments
        if self.tf_agents:
            self.obs_state = self._obs_preprocessing(np.zeros(shape=(self.obs_channels,
                                                                    self.obs_height,
                                                                    self.obs_width)))
        # used with stablebaselines HER to train more efficiently                                                            
        elif self.goal_env:
            self.obs_state = {'observation':np.zeros(shape=(self.obs_channels,
                                                            self.obs_height,
                                                            self.obs_width)),
                              'achieved_goal':np.zeros(shape=(self.obs_channels,
                                                            self.obs_height,
                                                            self.obs_width)),
                              'desired_goal':np.zeros(shape=(self.obs_channels,
                                                            self.obs_height,
                                                            self.obs_width))}
        # used for general GYM scenario
        else:
            self.obs_state = self._obs_preprocessing(np.zeros(shape=(self.obs_channels,
                                                        self.obs_height,
                                                        self.obs_width)))

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
        if self.tf_agents:
            self.observation_space = spaces.Box(low=0, high=29, shape=
                        ((self.obs_channels*self.obs_height*self.obs_width),), dtype=np.uint8)
        elif self.goal_env:
            self.observation_space = spaces.Dict({'observation':spaces.Box(low=0, high=29, shape=
                                        (self.obs_channels,self.obs_height,self.obs_width), dtype=np.uint8),
                                        'achieved_goal':spaces.Box(low=0, high=29, shape=
                                        (self.obs_channels,self.obs_height,self.obs_width), dtype=np.uint8),
                                        'desired_goal':spaces.Box(low=0, high=29, shape=
                                        (self.obs_channels,self.obs_height,self.obs_width), dtype=np.uint8)})
        else:
            self.observation_space = spaces.Box(low=0, high=29, shape=
                        (self.obs_channels,self.obs_height,self.obs_width), dtype=np.uint8)

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
        if self.tf_agents:
            self.obs_state = self._obs_preprocessing(np.zeros(shape=(self.obs_channels,
                                                                    self.obs_height,
                                                                    self.obs_width)))
        # used with stablebaselines HER to train more efficiently                                                            
        elif self.goal_env:
            self.obs_state = {'observation':np.zeros(shape=(self.obs_channels,
                                                            self.obs_height,
                                                            self.obs_width)),
                              'achieved_goal':np.zeros(shape=(self.obs_channels,
                                                            self.obs_height,
                                                            self.obs_width)),
                              'desired_goal':np.zeros(shape=(self.obs_channels,
                                                            self.obs_height,
                                                            self.obs_width))}
        # used for general GYM scenario
        else:
            self.obs_state = self._obs_preprocessing(np.zeros(shape=(self.obs_channels,
                                                        self.obs_height,
                                                        self.obs_width)))
        self.current_step = 0
        self.total_reward = 0
        self.current_word = str(np.random.choice(self.game_words))
        self.current_word_vec = [ord(char) - 96 for char in self.current_word]
        print(f"Correct word: {self.current_word}")
        return self.obs_state


    def render(self, mode='human', close=False):
    # Render the environment to the screen
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

        if self.tf_agents:
            self.obs_state[self._convert_slice_dim(a=0,b=self.current_step)] = action_vec
            self.obs_state[self._convert_slice_dim(a=1,b=self.current_step)] = result_vec
        elif self.goal_env:
            self.obs_state['observation'][0,self.current_step,:] = action_vec
            self.obs_state['observation'][1,self.current_step,:] = result_vec

            self.obs_state['achieved_goal'][0,self.current_step,:] = action_vec
            self.obs_state['achieved_goal'][1,self.current_step,:] = result_vec

            self.obs_state['desired_goal'][0,self.current_step,:] = self.current_word_vec
            self.obs_state['desired_goal'][1,self.current_step,:] = np.array([29,29,29,29,29])
        else:
            self.obs_state[0,self.current_step,:] = action_vec
            self.obs_state[1,self.current_step,:] = result_vec
        # self.obs_state[0,self.current_step,:] = action_vec
        # self.obs_state[1,self.current_step,:] = result_vec


    def _calc_reward(self):
        reward = 0
        win = True
        if self.tf_agents:
            result = self.obs_state[self._convert_slice_dim(a=1,b=self.current_step)]
            answer_letters = self.obs_state[self._convert_slice_dim(a=0,b=self.current_step)]
        elif self.goal_env:
            result = self.obs_state['observation'][1,self.current_step,:]
            answer_letters = self.obs_state['observation'][0,self.current_step,:]  
        else:
            result = self.obs_state[1,self.current_step,:]
            answer_letters = self.obs_state[0,self.current_step,:]            
        used_letters = []
        if self.current_step > 0:
            if self.tf_agents:
                if any([np.array_equal(self.obs_state[self._convert_slice_dim(a=0,b=self.current_step)],previous_word) for previous_word in self.obs_state[self._convert_slice_dim(a=0,b_max=self.current_step)]]):
                    return(self.penalty)
            elif self.goal_env:
                 if any([np.array_equal(self.obs_state['observation'][0,self.current_step,:],previous_word) for previous_word in self.obs_state['observation'][0,:self.current_step,:]]):
                    return(self.penalty)               
            else:
                if any([np.array_equal(self.obs_state[0,self.current_step,:],previous_word) for previous_word in self.obs_state[0,:self.current_step,:]]):
                    return(self.penalty)
        for answer in zip(result,answer_letters):
            if answer[0] == 28:
                win = False
                if answer[1] not in used_letters:
                    reward += self.right_letter_reward
                    used_letters.append(answer[1])
            elif answer[0]== 29:
                reward += self.right_position_reward
            else:
                win = False
        if win:
            reward = round(self.win_reward*(1/(self.current_step+1)))
        return reward


    def _check_done(self):
        if self.current_step==6:
            return True
        if self.tf_agents:
            obs_state = self.obs_state[self._convert_slice_dim(a=1)].reshape((self.obs_height,self.obs_width))
        elif self.goal_env:
            obs_state = self.obs_state['observation'][1,:,:]
        else:
            obs_state = self.obs_state[1,:,:]
        for vec in obs_state:
            for num in vec:
                if num!=29:
                    break
            else:
                return True
        return False


    def _get_words(self,word_file_path,simplify=False):
        with open(word_file_path,"r") as file:
            word_list = file.read().splitlines()

        if simplify:
            word_list = random.sample(word_list,20)
        return word_list


    def _obs_preprocessing(self,obs):
        return obs.flatten()


    def _convert_slice_dim(self,a=None,b=None,c=None,b_max=None):
        if a!=None and b_max!=None and c==None:
            start = 0
            stop = (a*self.obs_height*self.obs_width)+(self.obs_width*b_max)+self.obs_width
        if a!=None and b!=None and c==None:
            start = (a*self.obs_height*self.obs_width)+(self.obs_width*b)
            stop = (a*self.obs_height*self.obs_width)+(self.obs_width*b)+self.obs_width
        elif a!=None and b==None and c==None:
            start = (a*self.obs_height*self.obs_width)
            stop = (a*self.obs_height*self.obs_width)+(self.obs_height*self.obs_width)
        else:
            raise Exception(f"cannot convert slice request for: {a},{b},{c}")
        
        return(np.arange(start,stop))