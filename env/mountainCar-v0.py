import gym
import numpy as np
from algorithms.DQN import DQN

env = gym.make("MountainCarContinuous-v0")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

dqn = DQN()

for episode in range(10000):
  obs = env.reset()
  episode_reward = 0
  while True:
    if(episode %100 ==0):
      env.render(); 

    action = dqn.choose_action(obs)
    next_obs, reward, done, info = env.step(action)

    # position, velocity = obs

    dqn.store_transition(obs, action, reward, next_obs)

    episode_reward += reward

    if dqn.memory_counter > 150:
      dqn.learn()
      if done:
          print('Ep: ', episode, '| Ep_r: ', round(episode_reward, 2))

    if done:
      break

    obs = next_obs