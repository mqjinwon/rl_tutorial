import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from setting import *


class OptionCritic(nn.Module):
    def __init__(self):
        super(OptionCritic, self).__init__()

        self.data =[]

        self.fc1 = nn.Linear(4,  256)

        # global pi
        self.global_pi = nn.Linear(256, 2)
        self.terminate = nn.Linear(256, 1)
        self.intra_option_policy = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr = IR)
    
    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
        

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + GAMMA * self.v(s_prime) * done
        delta = td_target - self.v(s)
        
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()  
        # loss1 = F.smooth_l1_loss(self.v(s), td_target.detach())
        # loss2 = -torch.log(pi_a) * delta.detach()

        # self.optimizer.zero_grad()
        # loss1.mean().backward()
        # self.optimizer.step()    

        # self.optimizer.zero_grad()
        # loss2.mean().backward()
        # self.optimizer.step()      



def main():
    env = gym.make('CartPole-v1')
    model = OptionCritic()    
    print_interval = 20
    score = 0.0

    for n_epi in range(10000):
        done = False
        s = env.reset()

        # choose w according to an epsilon-soft policy over options

        while not done:
            # choose a according to pi(a|s)
            prob = model.pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample().item()

            #1. Options evalution:


            
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()

    
    pass

if __name__ == '__main__':
    main()