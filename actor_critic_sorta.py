import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.distributions import Categorical

import numpy as np
import matplotlib.pyplot as plt

from statistics import stdev, mean

import gym

from model import Network
from utils import set_seed


def plot(x):
    plt.plot(x)
    plt.show()


def update(model, state_values, log_probs, rewards, gamma=0.9, lam=0.95,
           eps=np.finfo(np.float32).eps.item()):
    R = 0
    returns = []
    policy_loss = []
    critic_loss = []

    # calculate return from each time step
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)

    rewards = torch.tensor(rewards)

    # normalize returns
    # agent does not converge (easily) otherwise
    returns = torch.tensor(returns)
    returns = (returns-returns.mean())/(returns.std()+eps)

    for i, ((log_prob, state_value), R) in enumerate(
        zip(zip(log_probs, state_values), returns)):
        # advantage
        advantage = R - state_value.item()
        # policy gradient
        policy_loss.append(-log_prob * advantage)
        # critic
        critic_loss.append(F.smooth_l1_loss(state_value, torch.tensor([R]).unsqueeze(0)))

    loss = torch.stack(policy_loss).sum() + torch.stack(critic_loss).sum()

    model['optim'].zero_grad()
    loss.backward()
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
        state_values, rewards, log_probs = [], [], []

        obs = env.reset()

        while True:
            obs = np.ravel(obs)    # flatten state for network
            obs = torch.from_numpy(obs).float().unsqueeze(0)
            action_probs, state_value = model['net'](obs)
            action, log_prob = select_action(action_probs)
            obs, reward, done, _ = env.step(action)

            state_values.append(state_value)
            rewards.append(reward)
            log_probs.append(log_prob)

            if done:
                break

        avg_returns.append(sum(rewards))
        update(model, state_values, log_probs, rewards)

        if episode % 10 == 0:
            print('Episode: {} - Episode Return: {} - Average Returns: {}'.format(
                episode, sum(rewards), mean(avg_returns)
            ))

        returns.append(mean(avg_returns))

    plot(returns)


def main(environment='CartPole-v0', n_episodes=500):
    env = gym.make('CartPole-v0')
    set_seed(env, 0)

    obs_shape = env.observation_space.shape
    obs_dim = obs_shape[0]
    n_actions = env.action_space.n

    model = {}

    model['net'] = Network(obs_dim, n_actions)

    optimizer = optim.Adam(model['net'].parameters(), lr=1e-2)
    model['optim'] = optimizer

    T_max = 400
    eta_min = 1e-4
    model['scheduler'] = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min)

    train(env, model)



torch.manual_seed(0)

if __name__ == '__main__':
    main()
