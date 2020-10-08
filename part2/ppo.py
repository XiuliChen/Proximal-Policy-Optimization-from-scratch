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

		self._init_hyperparameters()

	def _init_hyperparameters(self):
		# We need to specify the total number of steps in learn()
		# when collecting the transition, we need to specify how many
		# transition per batch
		self.timesteps_per_batch=3000
		self.max_timesteps_per_episode=1000

	def collect_experience(self):
		# collect a batch of data
		batch_obs=[] # num_step_per_batch x obs_dim
		batch_acts=[] # num_step_per_batch x act_dim
		batch_log_probs=[] # num_step_per_batch
		batch_rews=[] # num_step_per_batch
		batch_rtgs=[] # num_step_per_batch
		batch_epi_lens=[] # episode length for each episode

		# We need to specify the total number of steps in learn()
		# when collecting the transition, we need to specify how many
		# transition per batch
		t=0 # number of steps run so far
		while t<self.timesteps_per_batch:
			obs=env.reset()
			done=False
			eps_rew=[]
			while not done:
				t+=1
				# choose an action by running the current policy
				action, log_prob=self.choose_action(obs)
				obs,rew,done,~=self.env.step()

				batch_obs.append(obs)
				batch_acts.append(action)
				batch_log_probs.append(log_prob)
				eps_rew.append(rew)

			# collect episode reward and length
			batch_rews.append(eps_rew)
			batch_epi_lens.append(eps_rew)


	def choose_action(self):
		pass

	def learn(self, total_num_steps):
		pass