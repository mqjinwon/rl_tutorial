import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np

from setting import *

class OptionCritic(nn.module):
    def __init__(self, n_state, n_action, option_num = 1):
        super(OptionCritic, self).__init__()

        self.option_num = option_num

        self.shared_layer = nn.Linear(n_state, 256)

        # policy over options
        self.policy_over_options = nn.Linear(256, option_num)

        # actor parameter
        self.intra_option_policy = []
        self.termination = []

        # critic parameter
        self.Q_omega = []
        self.Q_u = []

        for i in range(option_num):
            self.intra_option_policy.append(nn.Linear(256, n_action))
            self.termination.append(nn.Linear(256, 1))
            self.Q_omega.append(nn.Linear(256, 1))
            self.Q_u.append(nn.Linear(256, 1))

        self.optimizer = optim.Adam(self.parameters(), lr = LR)
    
    ########################
    ##### model define #####
    ########################

    def policy_over_option(self, x):
        x = F.relu(self.shared_layer(x))
        x = self.policy_over_options(x)
        prob = F.softmax(x)

        return prob

    def intra_option_policy(self, x, option_num=0):
        x = F.relu(self.shared_layer(x))
        x = self.intra_option_policy[option_num](x)
        prob = F.softmax(x)
        
        return prob

    def termiation(self, x):
        x = F.relu(self.shared_layer(x))
        x = self.termination(x)
        prob = F.softmax(x)

        return prob

    def Q_omega(self, x, option_num=0):
        x = F.relu(self.shared_layer(x))
        x = self.Q_omega[option_num](x)

        return x

    def Q_u(self, x, option_num=0):
        x = F.relu(self.shared_layer(x))
        x = self.Q_u[option_num](x)

        return x

    #############################
    ##### Policy evaluation #####
    #############################

    def cache(self, state, option, action):
        self.last_state = state
        self.last_option = option
        self.last_action = action
        self.last_Q_omega = self.Q_omega[option](state, option)
        pass

    def policy_evaluation(self, state, option, action, reward, done, termination):
		# One step target for Q_omega
        target = reward - self.Q_omega[option](self.last_state, self.last_option)

        if not done:
            # get best other option
            best_other_option = None

            for i in range(self.option_num):
                if self.option_num is option:
                    continue
                if best_other_option is None:
                    best_other_option = self.Q_Omega[i](state)
                else:
                    tmp_Q_omega = self.Q_omega[i](state);
                    if best_other_option < tmp_Q_omega:
                        best_other_option = tmp_Q_omega

            # get target
            beta_omega = self.termiation[self.last_option](state)  
            target += GAMMA * ((1.0 - beta_omega)*self.Q_Omega[option](state, self.last_option) + beta_omega*best_other_option)

        # evaluation
        tderror_Q_Omega = F.smooth_l1_loss(target.detach(), self.last_Q_Omega)
        tderror_Q_U = F.smooth_l1_loss(target.detach(), self.Q_U[self.last_option](self.last_state, self.last_option, self.last_action))

        loss = tderror_Q_Omega + tderror_Q_U

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()  

    ##############################
    ##### Policy improvement #####
    ##############################

# class OptionCritic(nn.Module):
    def __init__(self, n_state, n_action, option_num = 1):
        super(OptionCritic, self).__init__()

        self.data =[]

        self.fc1 = nn.Linear(n_state, 256)

        # option num
        self.fc_pi = []
        self.fc_v = []
        for i in range(option_num):
            self.fc_pi.append(nn.Linear(256, n_action))
            self.fc_v.append(nn.Linear(256, 1))
        self.optimizer = optim.Adam(self.parameters(), lr = LR)
    
    def pi(self, x, option_num = 0, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi[option_num](x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
        

    def v(self, x, option_num = 0):
        x = F.relu(self.fc1(x))
        v = self.fc_v[option_num](x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_list, a_list, r_list, s_prime_list, done_list = [], [], [], [], []
        
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_list.append(s)
            a_list.append([a])
            r_list.append([r/100])
            s_prime_list.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_list.append([done_mask])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_list, dtype=torch.float), torch.tensor(a_list), \
                                                            torch.tensor(r_list, dtype=torch.float), torch.tensor(s_prime_list, dtype=torch.float), \
                                                            torch.tensor(done_list, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
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

    nstates = env.observation_space.shape[0]
    nactions = env.action_space.n # env.observation_space.shape[0]

    model = OptionCritic(nstates, nactions,4)    

    for n_epi in range(10000):
        done = False
        s = env.reset()

        prob = model.policy_over_option(s)
        m = Categorical(prob)
        option = m.sample().item()

        prob = model.intra_option_policy(torch.from_numpy(s).float(), option)
        m = Categorical(prob)
        a = m.sample().item()

        model.cache(s, option, a)

        # choose w according to an epsilon-soft policy over options
        while not done:

            # choose a according to policy_over_option
            prob = model.intra_option_policy(torch.from_numpy(s).float(), option)
            m = Categorical(prob)
            a = m.sample().item()

            s, r, done, info = env.step(a)
            
            # if termination in s_prime
            # choose new w according to an epsilon-soft policy over options
            if option_terminations[option].sample(state):
                option = policy_over_options.sample(state)

            # 1. option evaluation

            # 2. option improvement




            env.render()
            
            if done:
                break                     
            
            

    env.close()

    
    pass

if __name__ == '__main__':
    main()