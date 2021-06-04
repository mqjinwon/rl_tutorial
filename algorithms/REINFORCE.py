import torch
from torch import nn
import torch.nn.functional as F
import random

from setting import *

class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(INPUT_SIZE, 10)
    self.fc2 = nn.Linear(10, 10)
    self.fc3 = nn.Linear(10, OUTPUT_SIZE)
    

  def forward(self, x):

    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    x = self.fc3(x)
    x = F.softmax(x, dim=0)

    return x



class REINFORCE(object):

  def __init__(self):
    self.model = Net()
    self.data =[]
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=IR)


  def make_action(self, obs):
    obs = torch.from_numpy(obs).float()
    out = self.model.forward(obs)
    return out

    # # greedy action
    # if random.random() <EPSILON:
    #   return torch.min(out, 1)

    # else:
    #   return torch.max(out, 1)


  def put_data(self, item):
    self.data.append(item)


  def learn(self):
    R = 0
    self.optimizer.zero_grad()
    for r, prob in self.data[::-1]:
        R = r + GAMMA * R
        loss = -torch.log(prob) * R
        loss.backward()
    self.optimizer.step()
    self.data = []