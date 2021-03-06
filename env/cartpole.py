import gym
import numpy as np
import torch
from algorithms.REINFORCE import REINFORCE
from torch.distributions import Categorical

import matplotlib.pyplot as plt


from setting import *

"""
Description:
    A pole is attached by an un-actuated joint to a cart, which moves along
    a frictionless track. The pendulum starts upright, and the goal is to
    prevent it from falling over by increasing and reducing the cart's
    velocity.
Source:
    This environment corresponds to the version of the cart-pole problem
    described by Barto, Sutton, and Anderson
Observation:
    Type: Box(4)
    Num     Observation               Min                     Max
    0       Cart Position             -4.8                    4.8
    1       Cart Velocity             -Inf                    Inf
    2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
    3       Pole Angular Velocity     -Inf                    Inf
Actions:
    Type: Discrete(2)
    Num   Action
    0     Push cart to the left
    1     Push cart to the right
    Note: The amount the velocity that is reduced or increased is not
    fixed; it depends on the angle the pole is pointing. This is because
    the center of gravity of the pole increases the amount of energy needed
    to move the cart underneath it
Reward:
    Reward is 1 for every step taken, including the termination step
Starting State:
    All observations are assigned a uniform random value in [-0.05..0.05]
Episode Termination:
    Pole Angle is more than 12 degrees.
    Cart Position is more than 2.4 (center of the cart reaches the edge of
    the display).
    Episode length is greater than 200.
    Solved Requirements:
    Considered solved when the average return is greater than or equal to
    195.0 over 100 consecutive trials.
"""

def main():
  env = gym.make('CartPole-v1')
  pi = REINFORCE()

  score = 0.0

  plot_x = []
  plot_y = []

  for i_episode in range(2000):
        obs = env.reset()
        done = False

        while not done:
            prob = pi.make_action(obs)
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, info = env.step(a.item())
            pi.put_data((r, prob[a]))
            obs = s_prime
            score += r

            if i_episode%100 == 0:
                env.render()

        if i_episode%100==0 and i_episode!=0:
            print("# of episode :{}, avg score : {}".format(i_episode, score))


        pi.learn()
        plot_x.append(i_episode)
        plot_y.append(score)
        score = 0.0


  plt.plot(plot_x, plot_y)
  plt.show()    



  env.close()

if __name__ == "__main__":
  main()