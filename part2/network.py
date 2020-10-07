import torch
from torch import nn
import torch.nn.functional as F 


class FeedforwardNN(nn.Module):
	def __init__(self, input_size, output_size):
		super(FeedforwardNN,self).__init__()
		num_hidden=128
		self.layer1=nn.Linear(input_size,num_hidden)
		self.layer2=nn.Linear(num_hidden,num_hidden)
		self.layer3=nn.Linear(num_hidden,output_size)

	def forward(self,obs):
		x=F.relu(self.layer1(obs))
		x=F.relu(self.layer2(x))
		x=self.layer3(x)

		return x