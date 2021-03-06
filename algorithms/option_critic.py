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

        self.state_num = n_state
        self.action_num = n_action
        self.option_num = option_num

        self.shared_layer = nn.Linear(n_state, 256)
        self.q_u_layer = nn.Linear(n_state+1, 256)
        print(self.q_u_layer)

        # actor parameter
        self.intra_option_policy = []
        self.termination = []

        # critic parameter
        self.q_u = []

        for i in range(option_num):
            self.intra_option_policy.append(nn.Linear(256, n_action)) # policy
            self.termination.append(nn.Linear(256, 1))  # terminate or not
            self.q_u.append(nn.Linear(256, 1))  # value

        self.optimizer = optim.Adam(self.parameters(), lr = LR)
        self.cache =[]
    
    ########################
    ##### model define #####
    ########################

    def policyOverOption(self, x):

        prob = None
        for i in range(self.option_num):

            q_omega_value = self.qOmega(x, i)
            if i == 0:
                prob = q_omega_value
            else:
                prob = torch.cat((prob, q_omega_value), 0)

        prob = F.softmax(prob, dim=0)

        # print("policyOverOption: ", prob)

        return prob

    def intraOptionPolicy(self, x, option_num=0):
        x = F.relu(self.shared_layer(x))
        x = self.intra_option_policy[option_num](x)
        prob = F.softmax(x, dim=0)
        
        return prob

    def terminationPolicy(self, x, option_num=0):
        x = F.relu(self.shared_layer(x))
        x = self.termination[option_num](x)
        prob = torch.sigmoid(x)

        return prob

    def qOmega(self, s, option_num=0):
        intra_option = self.intraOptionPolicy(s, option_num)

        qu = None
        for i in range(self.action_num):
            tmp_qu = self.qU(s, torch.tensor([i], dtype=torch.float), option_num)
            if qu == None:
                qu = tmp_qu
            else:
                qu = torch.cat((qu, tmp_qu), 0)

        x = torch.dot(intra_option, qu)

        # print("intra_option: ", intra_option)
        # print("qu: ", qu)

        return torch.unsqueeze(x, 0)

    def qU(self, s, a, option_num=0):
        x = torch.cat((s,a), 0)
        x = self.q_u_layer(x)
        x = self.q_u[option_num](x)

        return x

    #############################
    ##### Policy evaluation #####
    #############################

    def policyEvaluation(self, s, a, r, s_prime, done, option):
		# One step target for Q_omega
        target = r - self.qU(s, a, option)

        if not done:
            # get max q_omega from other options
            best_other_q_omega = None

            for i in range(self.option_num):
                if best_other_q_omega is None:
                    best_other_q_omega = self.qOmega(s_prime, option)
                else:
                    tmp_Q_omega = self.qOmega(s_prime, i)
                    if best_other_q_omega < tmp_Q_omega:
                        best_other_q_omega = tmp_Q_omega

            # get target
            beta_omega = self.terminationPolicy(s_prime, option)
            # print(beta_omega)
            target += GAMMA * ((1.0 - beta_omega)*self.qOmega(s_prime, option) + beta_omega*best_other_q_omega)

        # evaluation
        loss = target

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()


    ##############################
    ##### Policy improvement #####
    ##############################

    def intraOptionPolicyImprovement(self, action_prob, s, a, option=0):
        loss = -torch.log(action_prob) * self.qU(s, a, option).detach()

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        pass

    def terminationPolicyImprovement(self, termin_prob, s_prime, option=0):
        best_other_q_omega = None

        for i in range(self.option_num):
            if best_other_q_omega is None:
                best_other_q_omega = self.qOmega(s_prime, option)
            else:
                tmp_Q_omega = self.qOmega(s_prime, i)
                if best_other_q_omega < tmp_Q_omega:
                    best_other_q_omega = tmp_Q_omega

        advantage = self.qOmega(s_prime, option) - best_other_q_omega
        loss = termin_prob * advantage

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

def main():

    env = gym.make('CartPole-v1')

    nstates = env.observation_space.shape[0]
    nactions = env.action_space.n # env.observation_space.shape[0]
    noptions = 2

    model = OptionCritic(nstates, nactions, noptions)

    print_interval = 20
    score = 0.0

    # each episode
    for n_epi in range(10000):

        # environment initialization
        done = False
        s = env.reset()

        # choose w according to an epsilon-soft policy over options
        prob = model.policyOverOption(torch.from_numpy(s).float())
        m = Categorical(prob)
        option = m.sample().item()

        score = 0

        # each step
        while not done:


            # choose a according to policy_over_option
            action_prob =  model.intraOptionPolicy(torch.from_numpy(s).float(), option)
            m = Categorical(action_prob)
            a = m.sample().item()

            s_prime, r, done, info = env.step(a)

            s = s_prime

            score += r

            # 1. option evaluation
            model.policyEvaluation(torch.from_numpy(s).float(), torch.tensor([a], dtype=torch.float), torch.tensor([r], dtype=torch.float), \
                                     torch.from_numpy(s_prime).float(), done, option)


            # if termination in s_prime
            # choose new w according to an epsilon-soft policy over options
            termin_prob = model.terminationPolicy(torch.from_numpy(s_prime).float(), option)
            terminate_flag = round(termin_prob.detach().numpy()[0])

            # 2. option improvement
            model.intraOptionPolicyImprovement(action_prob, torch.from_numpy(s).float(), torch.tensor([a], dtype=torch.float), option)
            model.terminationPolicyImprovement(termin_prob, torch.from_numpy(s_prime).float(), option)

            # if terminate
            if terminate_flag == 1:
                prob = model.policyOverOption(torch.from_numpy(s_prime).float())
                m = Categorical(prob)
                option = m.sample().item()
                # print("opion change to:", option)
            
            if done:
                break   

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()