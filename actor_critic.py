import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.distributions import Categorical

import numpy as np
import matplotlib.pyplot as plt

from statistics import stdev, mean
import multiprocessing

import gym

from model import Network
from utils import set_seed


def plot(x):
    plt.plot(x)
    plt.show()


def update(model, state_value, next_state_value, log_prob, reward):
    # critic loss: td error
    critic_loss = torch.tensor(reward) + next_state_value - state_value
    # policy loss
    policy_loss = -log_prob * critic_loss

    model['optim'].zero_grad()
    (policy_loss + critic_loss).backward(retain_graph=True)
    model['optim'].step()
    model['scheduler'].step()


def select_action(action_probs):
    # multinomial over actions
    m = Categorical(action_probs)
    action = m.sample()
    return action.item(), m.log_prob(action) 


def train(env, model, n_episodes=200):
    avg_returns = []
    returns = []

    for episode in range(1, n_episodes+1):
        rewards = []

        obs = env.reset()
        obs = torch.from_numpy(obs).float().unsqueeze(0)
        action_probs, state_value = model['net'](obs)

        next_state_value = 0

        while True:
            action, log_prob = select_action(action_probs)
            next_obs, reward, done, _ = env.step(action)

            next_obs = torch.from_numpy(next_obs).float().unsqueeze(0)
            action_probs, next_state_value = model['net'](next_obs)

            update(model, state_value, next_state_value, log_prob, reward)

            if done:
                rewards.append(0)
                break

            obs = next_obs
            state_value = next_state_value

            rewards.append(reward)            

        avg_returns.append(sum(rewards))

        if episode % 10 == 0:
            print('Episode: {} - Episode Return: {} - Average Returns: {}'.format(
                episode, sum(rewards), mean(avg_returns)
            ))

        returns.append(mean(avg_returns))

    plot(returns)


def main(environment='CartPole-v0', n_episodes=200):
    env = gym.make('CartPole-v0')
    set_seed(env, 0)

    obs_shape = env.observation_space.shape
    obs_dim = obs_shape[0]
    n_actions = env.action_space.n

    model = {}

    model['net'] = Network(obs_dim, n_actions, hidden_dim=[128], n_layers=2)

    optimizer = optim.Adam(model['net'].parameters(), lr=1e-3)
    model['optim'] = optimizer

    T_max = 400
    eta_min = 1e-5
    model['scheduler'] = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min)

    train(env, model, n_episodes)



torch.manual_seed(0)

if __name__ == '__main__':
    main()


# # creating processes 
#     p1 = multiprocessing.Process(target=print_square, args=(10, )) 
#     p2 = multiprocessing.Process(target=print_cube, args=(10, )) 
  
#     # starting process 1 
#     p1.start() 
#     # starting process 2 
#     p2.start() 
  
#     # wait until process 1 is finished 
#     p1.join() 
#     # wait until process 2 is finished 
#     p2.join() 