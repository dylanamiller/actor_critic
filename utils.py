import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

import torch


def plot(ep_return, avg_returns, stds, fig, ax):
    episode = np.arange(len(avg_returns))
    avg_returns = np.array(avg_returns)
    stds = np.array(stds)
    
    ax.clear()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Returns')
    
    # plot average returns
    ax.plot(episode, avg_returns, label='Average Returns')
    # plot standard deviations
    ax.fill_between(episode, avg_returns-stds[1:], avg_returns+stds[1:], 
                    facecolor='blue', alpha=0.1)
    ax.plot(episode, ep_return, label='Episode Return')
    
    ax.set_title('Returns')
    ax.legend()
    fig.tight_layout()        
    fig.canvas.draw()
    plt.show() 
    
    
def set_seed(env, seed=0):
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    
def normalize(X, eps=1e-8):
    return (X-X.mean())/(X.std()+eps)
    
    
class Trajectory:
    def __init__(self, n_traj=10, rollout=20, gamma=0.9, lam=0.95):
        self.n_traj = n_traj
        self.rollout = rollout
        
        self.gamma = gamma
        self.lam = lam
        
        self.batch = {'state_values': [], 
                      'rewards': [],
                      'log_probs': [],
                      'observations': [],
                      'dones': []}
        
        self.step_count = 0
        
        
    def add(self, state_values, rewards, log_probs, observations, dones):
        for key in self.batch.keys():
            self.batch[key].append(eval(key))
            
        self.step_count += 1
        
       
    def batch_full(self):
        if self.step_count // self.rollout == self.n_traj:
            return True
        else:
            return False
    
    
    def discount(self, delta):
        return scipy.signal.lfilter([1], [1, -self.gamma*self.lam], delta.detach().numpy()[::-1], axis=0)[::-1]
    
    
    def gae(self, rewards, state_values, dones):
        advantages = self.discount(rewards[:-1] + self.gamma*state_values[1:]*(1 - dones[1:]) - state_values[:-1])
        advantages = torch.tensor(normalize(advantages), dtype=torch.float32)   
        return torch.cat([advantages, torch.tensor([0], dtype=torch.float32).unsqueeze(0)])   
    
        
    def fetch(self, traj):
        R = 0
        returns = []
        
        # calculate return from each time step ('reward-to-go')
        for r in self.batch['rewards'][traj:traj+self.rollout][::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        # convert lists of required values into tensors
        returns = torch.tensor(returns).unsqueeze(1)
        state_values = torch.cat(self.batch['state_values'][traj:traj+self.rollout])
        log_probs = torch.cat(self.batch['log_probs'][traj:traj+self.rollout]).unsqueeze(1)
        dones = torch.tensor(self.batch['dones'][traj:traj+self.rollout]).unsqueeze(1).type(torch.FloatTensor)
        
        # calculate GAE, normalize advantage estimate, and set to correct size
        advantages = self.gae(returns, state_values, dones)
                                          
        returns = normalize(returns)
        
        return returns, advantages, log_probs
    
    
    def get_obs(self):
        obs_idx = torch.multinomial(torch.arange(self.step_count, dtype=torch.float32), self.rollout)
        return torch.cat(self.batch['observations'])[obs_idx, :]

    
    def clear(self):
        for key in self.batch.keys():
            del self.batch[key][:]
            
        self.step_count = 0
      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        