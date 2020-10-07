import gym

from ppo import PPO
from network import FeedForwardNN


# Similar format as stable baselines
# https://stable-baselines.readthedocs.io/en/master/guide/quickstart.html

# set env
env=gym.make('Pendulum-v0')

# set RL method
model=PPO(env=env)

# start training
model.learn(total_timesteps=100000)