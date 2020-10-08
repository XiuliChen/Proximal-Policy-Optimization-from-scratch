'''
Step1: define our initial policy (actor); and value function (critic)
	actor: input: obs; output: action
	critic: input: obs; output: value 
'''
from network import FeedforwardNN
import torch
from torch.distributions import MultivariateNormal


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

		# for Diagonal gaussian policy (actor network)
		# The actor network output a “mean” action on a forward pass, 
		# then create a covariance matrix with some standard deviation along the diagonals. 
		# Then, we can use this mean and stddev to generate a Multivariate Normal Distribution 
		# using PyTorch’s distributions
		self.cov_var=torch.full(size=(self.action_dim,),fill_value=self.covariance_std)
		self.cov_mat=torch.diag(self.cov_var) # the covariance matrix

	def _init_hyperparameters(self):
		# We need to specify the total number of steps in learn()
		# when collecting the transition, we need to specify how many
		# transition per batch
		self.timesteps_per_batch=100
		self.max_timesteps_per_episode=5
		# for diagonal gaussian policy (choose 0.5 arbitraiily)
		self.covariance_std=0.5

	def collect_experience(self):
		# collect a batch of data (shape)
		batch_obs=[] # num_step_per_batch x obs_dim
		batch_acts=[] # num_step_per_batch x act_dim
		batch_log_probs=[] # num_step_per_batch
		batch_rews=[] # num_step_per_episode x num of episodes
		batch_rtgs=[] # num_step_per_batch
		batch_eps_lens=[] # num of episodes 

		# We need to specify the total number of steps in learn()
		# when collecting the transition, we need to specify how many
		# transition per batch
		t=0 # number of steps run so far
		while t<self.timesteps_per_batch:
			obs=self.env.reset()
			done=False
			eps_rew=[]
			for eps_t in range(self.max_timesteps_per_episode):
				t+=1
				# choose an action by running the current policy
				action, log_prob=self.choose_action(obs)
				obs,rew,done,_ =self.env.step(action)

				batch_obs.append(obs)
				batch_acts.append(action)
				batch_log_probs.append(log_prob)
				eps_rew.append(rew)
				if done:
					break

			# collect episode reward and length
			batch_rews.append(eps_rew)
			batch_eps_lens.append(eps_t+1)
		print(batch_eps_lens)


	def choose_action(self,obs):
		'''
		Two most common kinds of stochastic policies in deep RL: categorical policies and 
		diagonal gaussian policies
		(1) categorical policies (for discrete action space): the final layer gives you logits 
		for each action, followed by a softmax to convert the logits into probabilities.
		(2) Diagonal gaussian policies (for continous action space): A neural network maps from
		observations to mean actions

		The actor network output a “mean” action on a forward pass, 
		then create a covariance matrix with some standard deviation along the diagonals. 
		Then, we can use this mean and stddev to generate a Multivariate Normal Distribution 
		using PyTorch’s distributions, 
		and then sample an action close to our mean. 
		We’ll also extract the log probability of that action in the distribution.
		'''
		mean=self.actor(obs)
		dist=MultivariateNormal(mean,self.cov_mat)
		# sample the action and get log probability

		action=dist.sample()
		log_prob=dist.log_prob(action)

		return action.detach().numpy(), log_prob.detach()
		# calling detach() since the action and log_prob  
  		# are tensors with computation graphs, so I want to get rid
  		# of the graph and just convert the action to numpy array.
 	 	# log prob as tensor is fine. Our computation graph will
  		# start later down the line.


	def learn(self, total_num_steps):
		pass