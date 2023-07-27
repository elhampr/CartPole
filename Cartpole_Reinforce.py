import gym
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import ptan
import torch.optim as optim

import numpy as np


ENV_NAME = "CartPole-v0"
HIDDEN_LAYER = 128
GAMMA = 0.99
LEARNING_RATE = 0.01
TRAIN_EPISODE = 4


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


#This function receives local reward per step for all steps within one completed episode
#and returns the discounted total reward for every step (local reward plus discounted future rewards)
def calc_episode_qval(episode_rewards):

    sum_r = 0.0
    disc_reward = []

    for r in reversed(episode_rewards):
        sum_r *= GAMMA
        sum_r += r

        disc_reward.append(sum_r)
    return list(reversed(disc_reward))    



if __name__ == "__main__":

    env = gym.make(ENV_NAME)
    writer = SummaryWriter(comment="-cartpole-reinforce")

    net = PolicyGradientNetwork(env.observation_space.shape[0], env.action_space.n)  
    #No need for target net

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    #This agent uses probablity to select action based on distributions from net
    ##change the output from float 64 (gym env type) to float 32 to take less memory
    ##apply softmax to convert net output to probablity; net outputs are raw scores or logits (not probability)
    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True) 
    
    #To know more about ExperienceSourceFirstLast: https://github.com/Shmuma/ptan/issues/17
    ##no need for replay buffer as PG is on-policy
    ##defalut steps per episode: steps_count=2
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA)      

    #step_idx = 0

    batch_episode = 0
    batch_state, batch_action, current_reward, batch_qval = [], [], [], []

    done_episode = 0
    total_reward = []

    for step_idx, exp in enumerate(exp_source):
        batch_state.append(exp.state)
        batch_action.append(int(exp.action))  #remember to use int
        current_reward.append(exp.reward)

        if exp.last_state is None:  #if the end of episode is reached
            batch_qval.extend(calc_episode_qval(current_reward))   #remember to use extend instead of append
            current_reward.clear()
            batch_episode += 1


        episode_reward = exp_source.pop_total_rewards()            #remember that is exp_source no exp!
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

        if batch_episode<TRAIN_EPISODE:
            continue  #go to line 74; ends current iteration in for loop

        optimizer.zero_grad()

        batch_state_tf = torch.FloatTensor(batch_state)
        batch_action_tf = torch.LongTensor(batch_action)
        batch_qval_tf = torch.FloatTensor(batch_qval)

        logits = net(batch_state_tf)
        log_prob = F.log_softmax(logits)
        log_prob_pg = batch_qval_tf*log_prob[range(len(batch_state)), batch_action_tf]  #for each input state we take the log probablity for the selected action
        
        loss = -log_prob_pg.mean()  
        loss.backward()
        optimizer.step()


        batch_episode = 0
        batch_state.clear()
        batch_action.clear()
        batch_qval.clear()


    writer.close()








        















        









