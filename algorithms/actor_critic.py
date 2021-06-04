import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
learning_rate = 0.0002
gamma = 0.98
n_rollout = 10

class ActorCritic(nn.Module):
    def __init__(self):
        pass
    
    def pi(self, x, softmax_dim=0):
        
        pass

    def v(self, x):
        pass

    def put_data(self, transition):
        pass

    def make_batch(self):
        pass

    def train_net(self):
        pass

def main():
    env = gym.make("CartPole-v1")
    model = ActorCritic()
    print_interval = 20
    score = 0.0

    for n_epi in range(10000):
        done = False
        s = env.reset()

        while not done:
            for t in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)
                model.put_data((s, a, r, s_prime, done))

                s = s_prime
                score += r

                env.render()

                if done:
                    break

            


    
    pass

if __name__ == '__main__':
    main()