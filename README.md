# wordle-rl
RL for Wordle

## Running the app

1) `docker-compose up` from the main project directory

2) Go to the jupyter lab url that is created

2) Run the `stablebaselines3_dqn_demo.ipynb` notebook to demo training a stable-baselines DQN agent on this Wordle environment.

## Tools I used

* OpenAI Gym
* stable-baselines3
* TF Agents

## Other tools that may be interesting

* Rllib (Ray RL library for distributed RL)
* Tensorforce (I guess like community supported take on TF Agents)
* Reverb (separate service to manage replay buffers; made by DeepMind)

## Resources and useful links

### Wordle related resources

* wordle word lists (https://gist.github.com/cfreshman)
Note: these word lists can also be taken from the javascript in the wordle website

* [Solving Wordle with information theory (youtube)](https://www.youtube.com/watch?v=v68zYyaEmEA)

### Tutorials for RL implementation

* [Making a custom openai gym env (stablebaselines tutorial)](https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html)
 
* [Making a custom openai gym env (blog post)](https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e)

 * [TF-Agents DQN Tutorial](https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial)

 * [TF Agents DQN Tutorial (youtube)](https://www.youtube.com/watch?v=2nKD6zFQ8xI)

### Resources for large discrete action space problem

* [RL in Large Discrete Action Spaces](https://arxiv.org/pdf/1512.07679.pdf)

* [Hindsight Experience Replay Buffers](https://arxiv.org/pdf/1707.01495.pdf)

* [Blog post explanation of HER buffer](https://towardsdatascience.com/reinforcement-learning-with-hindsight-experience-replay-1fee5704f2f8)

* [Soft Actor-Critic (paper)](https://arxiv.org/pdf/1801.01290.pdf)

* [OpenAI Spinning Up: Soft Actor-Critic](https://spinningup.openai.com/en/latest/algorithms/sac.html#)

* [SOFT ACTOR-CRITIC FOR DISCRETE ACTION SETTINGS (paper)](https://arxiv.org/pdf/1910.07207.pdf)

* [Adapting Soft Actor Critic for Discrete Action Spaces (blog)](https://towardsdatascience.com/adapting-soft-actor-critic-for-discrete-action-spaces-a20614d4a50a)

### General RL knowledge

* [Intro to RL textbook](http://incompleteideas.net/book/RLbook2020.pdf)

