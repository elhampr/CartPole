import gym
import ptan
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99
LEARN_RATE = 0.01
EPISODES = 4

class PolicyGradientNet(nn.Module):
    def __init__(self, obs_n, act_n):
        super(PolicyGradientNet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_n, 128),
            nn.ReLU(),
            nn.Linear(128, act_n)
            )
        
    def forward(self, x):
        return self.net(x)
        
def calc_qvals(rewards):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment = "-cartpole-reinforce")
    
    net = PolicyGradientNet(env.observation_space.shape[0], env.action_space.n)
    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor,\
                                   apply_softmax=True)
    #Second term above is for the fact that gym returns observation in float64 insead
    #of float32 accepted by pytorch
    #Enabling fotmax maked the outcome numbers to be in form of probablities
    
    #returns transitions: stat, action, local reward
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma = GAMMA)
    optimizer = optim.Adam(net.parameters(), lr = LEARN_RATE)

    #reporting params:
    total_reward = []
    episode_count = 0
    
    #training params:
    batch_episode = 0
    curr_reward = []
    batch_stat, batch_act, batch_qval = [], [], []
    
    for i, item in enumerate(exp_source):
        batch_stat.append(item.state)
        batch_act.append(int(item.action))
        curr_reward.append(item.reward)
        
        if item.last_state is None:
            batch_qval.extend(calc_qvals(curr_reward))
            curr_reward.clear()
            batch_episode += 1
            
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            episode_count += 1
            reward = new_rewards[0]
            total_reward.append(reward)
            mean_rewards = float(np.mean(total_reward[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                i, reward, mean_rewards, episode_count))
            writer.add_scalar("reward", reward, i)
            writer.add_scalar("reward_100", mean_rewards, i)
            writer.add_scalar("episodes", episode_count, i)
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!" % (i, episode_count))
                break       
     
        if batch_episode < EPISODES:
            continue
        
        optimizer.zero_grad()
        stat_v = torch.FloatTensor(batch_stat)
        act_v = torch.LongTensor(batch_act)
        qval_v = torch.FloatTensor(batch_qval)
        
        logits_v = net(stat_v)
        log_prob_v = F.log_softmax(logits_v)
        log_prob_act_v = qval_v*log_prob_v[range(len(batch_stat)), act_v]
        loss_v = -log_prob_act_v.mean()
        
        loss_v.backward()
        optimizer.step()
        
        batch_episode = 0
        batch_stat.clear()
        batch_act.clear()
        batch_qval.clear()
        
        
    writer.close()       
        
        
        
    
    
    
    
    
    
        

