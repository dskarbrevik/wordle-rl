# wordle-rl
RL for Wordle

## Running the app

1) `docker-compose up` from the main project directory

2) Go to the jupyter lab url that is created

2) Run the `stablebaselines3_dqn_demo.ipynb` notebook to demo training a stable-baselines DQN agent on this Wordle environment.


## Resources

Tutorials and other resources I used to make this project...

* wordle word lists (https://gist.github.com/cfreshman)
Note: these word lists can also be taken from the javascript in the wordle website

* Making a custom Gym environment
 https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html
 https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e

 * TF-Agents Tutorials
 https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
 https://www.youtube.com/watch?v=2nKD6zFQ8xI

* [RL in Large Discrete Action Spaces](https://arxiv.org/pdf/1512.07679.pdf)

* [Hindsight Experience Replay Buffers](https://arxiv.org/pdf/1707.01495.pdf)

* [Intro to RL textbook](http://incompleteideas.net/book/RLbook2020.pdf)
