import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np

from setting import *

class OptionCritic(nn.Module):
    def __init__(self, n_state, n_action, option_num = 1):
        super(OptionCritic, self).__init__()

        self.option_num = option_num

        self.shared_layer = nn.Linear(n_state, 256)

        # actor parameter
        self.intra_option_policy = []
        self.termination = []

        # critic parameter
        self.q_omega = nn.Linear(256, option_num)
        self.q_u = []

        for i in range(option_num):
            self.intra_option_policy.append(nn.Linear(256, n_action)) # policy
            self.termination.append(nn.Linear(256, 2))  # terminate or not
            self.q_u.append(nn.Linear(256, 1))  # value

        self.optimizer = optim.Adam(self.parameters(), lr = LR)
    
    ######################
    ##### model define #####
    ######################

    def policyOverOption(self, x):
        x = self.qOmega(x)
        prob = F.softmax(x, dim=0)

        return prob

    def intraOptionPolicy(self, x, option_num=0):
        x = F.relu(self.shared_layer(x))
        x = self.intra_option_policy[option_num](x)
        prob = F.softmax(x, dim=0)
        
        return prob

    def terminationPolicy(self, x , option_num=0):
        x = F.relu(self.shared_layer(x))
        x = self.termination[option_num](x)
        prob = F.softmax(x, dim=0)

        return prob

    def qOmega(self, x):
        x = self.shared_layer(x)
        x = self.q_omega(x)

        return x

    def qU(self, x, option_num=0):
        x = self.shared_layer(x)
        x = self.q_u[option_num](x)

        return x

    #########################
    ##### Policy evaluation #####
    #########################

    def policyEvaluation(self, s, a, r, s_prime, done, option, termination):
		# One step target for Q_omega
        target = r - self.qOmega(self.last_state, self.last_option)

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

        # def update_Qs(self, state, option, action, reward, done, terminations):
        #     # One step target for Q_Omega
        #     target = reward
        #     if not done:
        #         beta_omega = terminations[self.last_option].pmf(state)
        #         target += self.discount * ((1.0 - beta_omega)*self.Q_Omega(state, self.last_option) + \
        #                     beta_omega*np.max(self.Q_Omega(state)))

        #     # Difference update
        #     tderror_Q_Omega = target - self.last_Q_Omega
        #     self.Q_Omega_table[self.last_state, self.last_option] += self.lr * tderror_Q_Omega

        #     tderror_Q_U = target - self.Q_U(self.last_state, self.last_option, self.last_action)
        #     self.Q_U_table[self.last_state, self.last_option, self.last_action] += self.lr * tderror_Q_U

        #     # Cache
        #     self.last_state = state
        #     self.last_option = option
        #     self.last_action = action
        #     if not done:
        #         self.last_Q_Omega = self.Q_Omega(state, option)


    ###########################
    ##### Policy improvement #####
    ###########################



def main():
    env = gym.make('CartPole-v1')

    nstates = env.observation_space.shape[0]
    nactions = env.action_space.n # env.observation_space.shape[0]

    model = OptionCritic(nstates, nactions, 2)    

    # each episode
    for n_epi in range(10000):
        done = False
        s = env.reset()

        # choose w according to an epsilon-soft policy over options
        prob = model.policyOverOption(torch.from_numpy(s).float())
        m = Categorical(prob)
        option = m.sample().item()

        # each step
        while not done:

            # choose a according to policy_over_option
            prob =  model.intraOptionPolicy(torch.from_numpy(s).float(), option)
            m = Categorical(prob)
            a = m.sample().item()

            s_prime, r, done, info = env.step(a)

            # choose new w according to an epsilon-soft policy over options
            prob = model.terminationPolicy(torch.from_numpy(s_prime).float(), option)
            m = Categorical(prob)
            terminate_flag = m.sample().item()

            # if terminate
            if terminate_flag is 1:
                prob = model.policyOverOption(torch.from_numpy(s_prime).float())
                m = Categorical(prob)
                option = m.sample().item()

            s = s_prime

            # 1. option evaluation

            # 2. option improvement

            # if termination in s_prime




            env.render()
            
            if done:
                break                     
            
    env.close()

    
    pass

if __name__ == '__main__':
    main()