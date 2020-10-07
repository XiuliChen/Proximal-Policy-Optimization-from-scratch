'''
Step1: define our initial policy (actor); and value function (critic)
	actor: input: obs; output: action
	critic: input: obs; output: value 
'''

def PPO():
	def __init__(self, env):
		# It is a openai gym env
		self.env=env
		# the observation and action dimension
		self.obs_dim=env.observation_space.shape[0]
		self.action_dim=env.action_space.shape[0]
		
		# For now, we use the simple Feed-Forward Neural Network for our actor/critic networks
		# In future versions, we will add an policy_type variable for the PPO class
		self.actor=FeedforwardNN(self.obs_dim,self.action_dim)
		self.critic=FeedforwardNN(self.obs_dim,1)

	def learn(self, total_num_steps):
		pass