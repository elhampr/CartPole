import gym
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import ptan
import torch.optim as optim

import numpy as np
from typing import Optional


ENV_NAME = "CartPole-v0"
HIDDEN_LAYER = 128

GAMMA = 0.99
LEARNING_RATE = 0.001
TRAIN_EPISODE = 8
REWARD_STEPS = 10   
ENTROPY_BETA = 0.01


class PolicyGradientNetwork(nn.Module):
    def __init__(self, input_size, action_size):
        super(PolicyGradientNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size,HIDDEN_LAYER),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER, action_size)
        )

    def forward(self, x):
        return self.net(x)    


def smooth(old: Optional[float], val: float, alpha: float = 0.95) -> float:
    if old is None:
        return val
    return old * alpha + (1-alpha)*val

#This function receives local reward per step for all steps within one completed episode
#and returns the discounted total reward for every step (local reward plus discounted future rewards)
# def calc_episode_qval(episode_rewards):

#     sum_r = 0.0
#     disc_reward = []

#     for r in reversed(episode_rewards):
#         sum_r *= GAMMA
#         sum_r += r

#         disc_reward.append(sum_r)
#     return list(reversed(disc_reward))    



if __name__ == "__main__":

    env = gym.make(ENV_NAME)
    writer = SummaryWriter(comment="-cartpole-pg")

    net = PolicyGradientNetwork(env.observation_space.shape[0], env.action_space.n)  
    #No need for target net

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    #This agent uses probablity to select action based on distributions from net
    ##change the output from float 64 (gym env type) to float 32 to take less memory
    ##apply softmax to convert net output to probablity; net outputs are raw scores or logits (not probability)
    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True) 
    
    #To know more about ExperienceSourceFirstLast: https://github.com/Shmuma/ptan/issues/17
    ##no need for replay buffer as PG is on-policy
    ##To address the full episode requirements issue in REINFORCE we increase the steps (to unroll bellman equation) from deafult 2 
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)      

    #step_idx = 0

    #batch_episode = 0  #replaced by len(batch_state)
    batch_state, batch_action, batch_scale, batch_qval = [], [], [], []

    done_episode = 0
    reward_sum = 0.0
    total_reward = []

    bs_smoothed = entropy = l_entropy = l_policy = l_total = None

    for step_idx, exp in enumerate(exp_source):
        #Below two lines compute the baseline value (mean of reward here) to subtract from Qs to address the high gradients variance issue in REINFORCE
        reward_sum += exp.reward
        baseline_to_subtract = reward_sum/(step_idx+1)   

        writer.add_scalar("baseline", baseline_to_subtract, step_idx)

        batch_state.append(exp.state)
        batch_action.append(int(exp.action))  #remember to use int
        batch_scale.append(exp.reward - baseline_to_subtract)   #Store scale instead of current reward


        ##No need for the below function since the experience already includes the discounted rewards every N steps
        # if exp.last_state is None:  #if the end of episode is reached
        #     batch_qval.extend(calc_episode_qval(current_reward))   #remember to use extend instead of append
        #     current_reward.clear()
        #     batch_episode += 1

        episode_reward = exp_source.pop_total_rewards()   
        if episode_reward:
            done_episode += 1
            total_reward.append(episode_reward[0])
            mean_rew = float(np.mean(total_reward[-100:]))
            #step index: num of steps progressed, reward:total reward of the current done episode
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" %(step_idx, episode_reward[0], mean_rew, done_episode))

            writer.add_scalar("reward", episode_reward[0], step_idx)
            writer.add_scalar("mean_reward_100", mean_rew, step_idx)
            writer.add_scalar("episodes", done_episode, step_idx)

            if mean_rew > 195:
                print("solved in %d steps and %d episodes" %(step_idx, done_episode))
                break

        if len(batch_state)<TRAIN_EPISODE:   #Here we replace the batch_episode counter with length function (no need to store a var)
            continue  #go to line 74; ends current iteration in for loop

        optimizer.zero_grad()

        batch_state_tf = torch.FloatTensor(batch_state)
        batch_action_tf = torch.LongTensor(batch_action)
        batch_scale_tf = torch.FloatTensor(batch_scale)

        logits = net(batch_state_tf)
        log_prob = F.log_softmax(logits, dim=1)
        #Reformulate the log prob. pg to have modified scale (reward-avg.reward) to address the high variance issue
        log_prob_pg = batch_scale_tf*log_prob[range(len(batch_state)), batch_action_tf]  #for each input state we take the log probablity for the selected action
        loss = -log_prob_pg.mean()                      #policy loss (negative sign)
        
        #We take additional steps to compute entropy bonus and subtract it from loss in order to address exploration issue in REINFORCE
        prob = F.softmax(logits, dim=1)
        entropy = -(prob*log_prob).sum(dim=1).mean()    #entropy loss (negative sign)
        loss_final = loss + ENTROPY_BETA*(-entropy)     #entropy again negative (since we want to subtract it from loss function)

        loss_final.backward()
        optimizer.step()

        #To evaluate the effect of the implemented strategies on the policy, we caluulate the difference between two policies
        #before and after the net optimization using KL-div
        new_prob = F.softmax(net(batch_state_tf), dim=1)
        kl_div = -((new_prob/prob).log()*prob).sum(dim=1).mean()
        writer.add_scalar("kl", kl_div.item(), step_idx)

        
        #Below we try to gather info about gradient behaviour 
        grad_max = 0.0
        grad_means = 0.0
        grad_count = 0
        for p in net.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means += (p.grad ** 2).mean().sqrt().item()
            grad_count += 1

        bs_smoothed = smooth(bs_smoothed, np.mean(batch_scale))
        entropy = smooth(entropy, entropy.item())
        l_entropy = smooth(l_entropy, -ENTROPY_BETA*(entropy.item()))
        l_policy = smooth(l_policy, loss.item())
        l_total = smooth(l_total, loss_final.item())

        writer.add_scalar("baseline", baseline_to_subtract, step_idx)
        writer.add_scalar("entropy", entropy, step_idx)
        writer.add_scalar("loss_entropy", l_entropy, step_idx)
        writer.add_scalar("loss_policy", l_policy, step_idx)
        writer.add_scalar("loss_total", l_total, step_idx)
        writer.add_scalar("grad_l2", grad_means / grad_count, step_idx)
        writer.add_scalar("grad_max", grad_max, step_idx)
        writer.add_scalar("batch_scales", bs_smoothed, step_idx)




        batch_state.clear()
        batch_action.clear()
        batch_scale.clear()


    writer.close()
