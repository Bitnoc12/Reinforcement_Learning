import gym
import torch
import pylab
import random
import argparse
import numpy as np
from collections import deque
from datetime import datetime
from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gym.wrappers import Monitor
from torch.nn import init

record_every = 100

class OrnsteinUhlenbeckActionNoise(object):
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    #根据给出的变量，写出OU噪声下采样值的公式，函数名为sample

    def sample(self):
        dx = self.theta * (self.mu - self.X) + self.sigma * np.random.randn(self.action_dim)
        self.X += dx
        return self.X


class Actor(nn.Module):
    #代码编写：根据下方主函数的参数引用，编写actor网络，并定义前项传播
    def __init__(self,state_dim,action_dim,action_bound):
        super(Actor,self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        num_hiddens = 300

        self.net = nn.Sequential(
            nn.Linear(self.state_dim,num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens,self.action_dim),
            nn.Tanh()
        )

        #权重初始化
        for name,params in self.net.named_parameters():
            if 'bias' in name:
                init.constant_(params,val=0.)
            else:
                init.normal_(params,mean=0,std=0.001)

    #定义网络的前向传播
    def forward(self,states):
        actions = self.net(states)
        scaled_actions = actions * self.action_bound
        return scaled_actions


class Critic(nn.Module):
    # 代码编写：根据下方主函数的参数引用，编写critic网络，并定义前项传播
    def __init__(self, state_dim, action_dim,action_bound):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        num_hiddens = 300

        self.params = nn.ParameterDict({
            'w1_s': nn.Parameter(torch.randn(self.state_dim,num_hiddens)*0.001),
            'w1_a': nn.Parameter(torch.randn(action_dim,num_hiddens)*0.001),
            'b1': nn.Parameter(torch.zeros(1,num_hiddens))
        })
        self.linear = nn.Linear(num_hiddens,1)

        # 权重初始化
        for name, params in self.linear.named_parameters():
            if 'bias' in name:
                init.constant_(params, val=0.)
            else:
                init.normal_(params, mean=0, std=0.001)

    # 定义网络的前向传播
    def forward(self, states,actions):
        y1 = torch.mm(states,self.params['w1_s'])
        y2 = torch.mm(actions,self.params['w1_a'])
        y = torch.relu(y1+y2+self.params['b1'])
        q = self.linear(y)
        return q




class DDPG(object):
    def __init__(self, options):
        # hyperparameter
        self.memory_size = options.get('memory_size', 1000000)
        self.action_size = options.get('action_size')
        self.action_range = options.get('action_range')
        self.obs_size = options.get('obs_size')
        self.batch_size = options.get('batch_size')
        self.actor_lr = options.get('actor_lr')
        self.critic_lr = options.get('critic_lr')
        self.gamma = options.get('gamma')
        self.decay = options.get('decay')
        self.tau = options.get('tau')

        # actor model
        self.actor = Actor(self.obs_size, self.action_size, self.action_range)
        self.actor_target = Actor(self.obs_size, self.action_size, self.action_range)

        # critic model
        self.critic = Critic(self.obs_size, self.action_size, self.action_range)
        self.critic_target = Critic(self.obs_size, self.action_size, self.action_range)

        # memory(uniformly)
        self.memory = deque(maxlen=self.memory_size)

        # explortion
        self.ou = OrnsteinUhlenbeckActionNoise(theta=args.ou_theta, sigma=args.ou_sigma,
                                               mu=args.ou_mu, action_dim=self.action_size)

        # optimizer
        #自行定义adam优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=self.critic_lr)


        # initialize target model
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def get_action(self, state):
        state = torch.from_numpy(state).float()
        model_action = self.actor(state).detach().numpy() * self.action_range
        action = model_action + self.ou.sample() * self.action_range
        return action

    def update_target_model(self):
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((deepcopy(state), action, reward, deepcopy(next_state), done))

    def _get_sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def save(self):
        torch.save(self.actor.state_dict(), 'actor.pth')
        torch.save(self.critic.state_dict(), 'critic.pth')
        print("====================================")
        print("model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load('actor.pth'))
        self.critic.load_state_dict(torch.load('critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


    def train(self):
        minibatch = np.array(self._get_sample(self.batch_size)).transpose()
        states = np.vstack(minibatch[0])
        actions = np.vstack(minibatch[1])
        rewards = np.vstack(minibatch[2])
        next_states = np.vstack(minibatch[3])
        dones = np.vstack(minibatch[4].astype(int))

        rewards = torch.Tensor(rewards)
        dones = torch.Tensor(dones)
        actions = torch.Tensor(actions)

        # critic update
        self.critic_optimizer.zero_grad()
        states = torch.Tensor(states)
        next_states = torch.Tensor(next_states)
        next_actions = self.actor_target(next_states)

        #写出损失计算和反向传播过程，用于优化critic网络
        target_q = rewards + (1-dones) * self.gamma * self.critic_target(next_states,next_actions)
        current_q = self.critic(states,actions)
        critic_loss = F.mse_loss(current_q,target_q)
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor update
        self.actor_optimizer.zero_grad()
        pred_actions = self.actor(states)

        actor_loss = -self.critic(states,pred_actions).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        # 写出损失计算和反向传播过程，用于优化actor网络


def main(args):
    env = gym.make(args.env)
    '''
    env = Monitor(env, directory="./monitor",
                  resume=True,
                  video_callable=lambda count: count % record_every == 0)
    '''
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_range = env.action_space.high[0]

    print(action_size, action_range)

    args_dict = vars(args)
    args_dict['action_size'] = action_size
    args_dict['obs_size'] = obs_size
    args_dict['action_range'] = action_range

    scores, episodes = [], []
    agent = DDPG(args_dict)
    recent_reward = deque(maxlen=100)
    frame = 0

    for e in range(args.episode):
        score = 0
        step = 0
        done = False
        state = env.reset()
        state = np.reshape(state, [1, agent.obs_size])
        while not done:
            step += 1
            frame += 1
            if args.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)

            next_state, reward, done, info = env.step([action])
            next_state = np.reshape(next_state, [1, agent.obs_size])

            reward = float(reward[0, 0])
            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)

            score += reward
            state = next_state
            if frame > agent.batch_size:
                agent.train()
                agent.update_target_model()

            if frame % 2000 == 0:
                print('now time : ', datetime.now())
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./pendulum_ddpg.png")

            if done:
                recent_reward.append(score)
                # every episode, plot the play time
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "   steps:", step,
                      "    recent reward:", np.mean(recent_reward))

                # if the mean of scores of last 10 episode is bigger than 400
                # stop training
        if e %1000==0:
            agent.save()

def test():
    env = gym.make(args.env)

    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_range = env.action_space.high[0]

    args_dict = vars(args)
    args_dict['action_size'] = action_size
    args_dict['obs_size'] = obs_size
    args_dict['action_range'] = action_range

    agent = DDPG(args_dict)
    agent.load()
    recent_reward = deque(maxlen=100)
    frame = 0
    test_iteration = 10

    for e in range(test_iteration):
        score = 0
        step = 0
        done = False
        state = env.reset()
        state = np.reshape(state, [1, agent.obs_size])
        while not done:
            step += 1
            frame += 1
            env.render()
            # get action for the current state and go one step in environment
            action = agent.get_action(state)

            next_state, reward, done, info = env.step([action])
            next_state = np.reshape(next_state, [1, agent.obs_size])

            reward = float(reward[0, 0])

            score += reward
            state = next_state
            if done:
                recent_reward.append(score)
                # every episode, plot the play time
                print("episode:", e, "  score:", score,  "   steps:", step,
                      "    recent reward:", np.mean(recent_reward))

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', default='Pendulum-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--episode', default=10000, type=int, help='the number of episode')
    parser.add_argument('--render', default=False, type=bool, help='is render')
    parser.add_argument('--memory_size', default=500000, type=int, help='replay memory size')
    parser.add_argument('--batch_size', default=64, type=int, help='minibatch size')
    parser.add_argument('--actor_lr', default=1e-4, type=float, help='actor learning rate')
    parser.add_argument('--critic_lr', default=1e-3, type=float, help='critic learning rate')
    parser.add_argument('--gamma', default=0.99, type=float, help='discounted factor')
    parser.add_argument('--decay', default=1e-2, type=int, help='critic weight decay')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')

    args = parser.parse_args()
    print(vars(args))
    # test()
    main(args)
    # test()