'''
Step1: define our initial policy (actor); and value function (critic)
	actor: input: obs; output: action
	critic: input: obs; output: value 
'''
from network import FeedforwardNN

class PPO:
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

	def collect_experience(self):
		# collect a batch of data
		batch_obs=[] # num_step_per_batch x obs_dim
		batch_acts=[] # num_step_per_batch x act_dim
		batch_log_probs=[] # num_step_per_batch
		batch_rews=[] # num_step_per_batch
		batch_rtgs=[] # num_step_per_batch
		batch_epi_lens=[] # episode length for each episode

	def learn(self, total_num_steps):
		pass