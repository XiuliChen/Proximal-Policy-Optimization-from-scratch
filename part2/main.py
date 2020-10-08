import gym

from ppo import PPO



# Similar format as stable baselines
# https://stable-baselines.readthedocs.io/en/master/guide/quickstart.html

# set env
env=gym.make('Pendulum-v0')

# set RL method
model=PPO(env)

model.collect_experience()

# start training
model.learn(total_num_steps=100000)