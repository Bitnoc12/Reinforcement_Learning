import os
import gym
import shutil
import argparse
import numpy as np
from tqdm import trange
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class PPOMemory:
    def __init__(self, gamma, tau):
        self.states = []
        self.actions = []
        self.rewards = []       #环境信息
        self.values = []        #值函数
        self.logprobs = []      #对数概率
        self.tdlamret = []      #advants + values
        self.advants = []       #优势函数
        self.gamma = gamma
        self.tau = tau
        self.ptr = 0                #数组索引
        self.path_start_idx = 0     #数组索引

    def store(self, s, a, r, v, lp):
        #请编写向经验池中添加经验的代码
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.values.append(v)
        self.logprobs.append(lp)

    def finish_path(self, v):
        #制作一批数据
        path_slice = np.arange(self.path_start_idx, self.ptr)
        rewards_np = np.array(self.rewards)[path_slice]
        values_np = np.array(self.values)[path_slice]
        values_np_added = np.append(values_np, v)
        gae = 0
        advants = []
        for t in reversed(range(len(rewards_np))):
            delta = rewards_np[t] + self.gamma * values_np_added[t+1] - values_np_added[t]
            gae = delta + self.gamma * self.tau * gae
            advants.insert(0, gae)
        self.advants.extend(advants)

        advants_np = np.array(advants)
        tdlamret_np = advants_np + values_np
        self.tdlamret.extend(tdlamret_np.tolist())
        self.path_start_idx = self.ptr

    def reset_storage(self):
        #请写出完成一轮mini-batch后清空经验的代码
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.logprobs = []

    def get(self):
        # 重置标记
        data = dict(states=self.states, actions=self.actions, logpas=self.logprobs,
                    rewards=self.rewards, values=self.values,
                    tdlamret=self.tdlamret, advants=self.advants)
        self.reset_storage()
        return data

    def __len__(self):
        return len(self.rewards)

class PPO():
    def __init__(self):
        super(PPO, self).__init__()
        self.seed = 66
        self.average_interval = 100
        self.gae_tau = 0.95                 #原论文给的合适的参数
        self.gamma = 0.99
        self.max_episodes = 50000
        self.max_steps_per_episode = 300
        self.batch_size = 32
        self.clip_range = 0.2               #CLIP参数
        self.coef_entpen = 0.001            #目标函数L中的熵S的参数c2
        self.coef_vf = 0.5                  #目标函数L中的VF的参数c1
        self.memory_size = 2048
        self.optim_epochs = 4
        self.terminal_score = 230           #最终目标得分
        self.evn_name = "LunarLander-v2"
        self.lr = 0.002
        self.betas = [0.9,0.999]
        self.game = gym.make(self.evn_name)
        self.input_dim = self.game.observation_space.shape[0]
        self.output_dim = self.game.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.actor = Actor(device=self.device,input_dim=self.input_dim, output_dim=self.output_dim,)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=self.lr,betas=self.betas)
        self.critic = Critic(device=self.device , input_dim=self.input_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=self.lr,betas=self.betas)
    @staticmethod
    def load_weight(model):
        #读取模型，此处为静态方法原因在于在本PPO类中会调用，只是类实例化之前的静态变量和函数等问题和RL无关
        weight = torch.load("ppo.pth")
        model.load_state_dict(weight)
    @staticmethod
    def save_weight(model):
        #保存模型和检查点(如果有)，此处为静态方法原因在于在本PPO类中会调用，只是类实例化之前的静态变量和函数等问题和RL无关
        weight = model.state_dict()
        torch.save(weight, "ppo.pth")
    # 训练
    def train(self):
        env = gym.make(self.evn_name)
        env.seed(self.seed)
        self.memory = PPOMemory(gamma=self.gamma, tau=self.gae_tau)
        score_queue = deque(maxlen=self.average_interval)
        length_queue = deque(maxlen=self.average_interval)

        for episode in trange(1, self.max_episodes + 1):
            self.episode = episode
            episode_score = 0
            state = env.reset()
            for t in range(1, self.max_steps_per_episode + 1):
                if episode % 100 == 0:
                    env.render()

                with torch.no_grad():
                    state_tensor = torch.tensor(state).unsqueeze(0).float()
                    action_tensor, log_prob_tensor = self.actor.select_action(state_tensor)
                    value_tensor = self.critic(state_tensor)

                action = action_tensor.item()
                log_prob = log_prob_tensor.item()
                value = value_tensor.item()

                next_state, reward, done, _ = env.step(action)
                self.memory.store(state, action, reward, value, log_prob)

                episode_score += reward

                state = next_state

                if done or t == self.max_steps_per_episode:
                    if done:
                        v = 0
                    else:
                        next_state_tensor = torch.tensor(next_state).unsqueeze(0).float()
                        with torch.no_grad():
                            next_value_tensor = self.critic(next_state_tensor)
                        v = next_value_tensor.item()

                    self.memory.finish_path(v)

                    if len(self.memory) == self.memory_size:
                        data = self.prepare_data(self.memory.get())
                        self.optimize_ppo(data)

                    if done:
                        score_queue.append(episode_score)
                        length_queue.append(t)
                        break

            avg_score = np.mean(score_queue)
            std_score = np.std(score_queue)
            avg_duration = np.mean(length_queue)
            if episode % 100 == 0:
                print("{} - score: {:.1f} +-{:.1f} \t duration: {}".format(episode, avg_score, std_score, avg_duration))
                print("found best model at episode: {}".format(episode))
                self.save_weight(self.actor)

        self.save_weight(self.actor)
        return avg_score
    # 优化
    def optimize(self):
        data = self.prepare_data(self.memory.get())
        self.optimize_ppo(data)

    def prepare_data(self, data):
        #从PPO-memory中获得一组数据，数据存放在data字典中，然后分别将他们拿出来并转换相应的类型
        states_tensor = torch.from_numpy(np.stack(data['states'])).float()
        actions_tensor = torch.tensor(data['actions']).long()
        logpas_tensor = torch.tensor(data['logpas']).float()
        tdlamret_tensor = torch.tensor(data['tdlamret']).float()
        advants_tensor = torch.tensor(data['advants']).float()
        values_tensor = torch.tensor(data['values']).float()

        # 经过加工处理数据分布会更好，有一定可能产生更好效果，但也可以不进行
        advants_tensor = (advants_tensor - advants_tensor.mean()) / (advants_tensor.std() + 1e-5)

        data_tensor = dict(states=states_tensor, actions=actions_tensor, logpas=logpas_tensor,
                    tdlamret=tdlamret_tensor, advants=advants_tensor, values=values_tensor)
        return data_tensor
    #建立一个循环的迭代器用于PPO训练，因为每一次循环前进的变量有些多使用zip或for可能有一些麻烦，此处使用此方法和RL无关
    def ppo_iter(self, batch_size, ob, ac, oldpas, atarg, tdlamret, vpredbefore):
        total_size = ob.size(0)
        indices = np.arange(total_size)
        np.random.shuffle(indices)
        n_batches = total_size // batch_size
        for nb in range(n_batches):
            ind = indices[batch_size * nb : batch_size * (nb+1)]
            yield ob[ind], ac[ind], oldpas[ind], atarg[ind], tdlamret[ind], vpredbefore[ind]

    def optimize_ppo(self, data):

        ob = data['states']
        ac = data['actions']
        oldpas = data['logpas']
        atarg = data['advants']
        tdlamret = data['tdlamret']
        vpredbefore = data['values']
        eps = self.clip_range

        policy_losses = []
        entropy_losses = []
        value_losses = []

        # 对每个 policy_update_epochs
        for i in range(self.optim_epochs):
            # 对每个 batch
            data_loader = self.ppo_iter(self.batch_size,
                                        ob, ac, oldpas, atarg, tdlamret, vpredbefore)
            for batch in data_loader:
                ob_b, ac_b, old_logpas_b, atarg_b, vtarg_b, old_vpred_b = batch

                ratio = torch.exp(self.actor(ob_b).gather(1, ac_b) - old_logpas_b)
                surr1 = ratio * atarg_b
                surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * atarg_b
                pol_surr = -torch.min(surr1, surr2).mean()

                # Clipped value loss
                value_clipped = old_vpred_b + torch.clamp(self.critic(ob_b) - old_vpred_b, -eps, eps)
                vf_losses1 = (self.critic(ob_b) - vtarg_b).pow(2)
                vf_losses2 = (value_clipped - vtarg_b).pow(2)
                vf_loss = 0.5 * torch.max(vf_losses1, vf_losses2).mean()

                # Entropy loss
                pol_entpen = -self.actor(ob_b).sum(dim=1).mean()

                total_loss = pol_surr - pol_entpen * self.coef_entpen + vf_loss * self.coef_vf

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                policy_losses.append(pol_surr.item())
                entropy_losses.append(pol_entpen.item())
                value_losses.append(vf_loss.item())

            # for batch in data_loader:
            #     ob_b, ac_b, old_logpas_b, atarg_b, vtarg_b, old_vpred_b = batch
            #     #计算新旧策略的比率
            #     _, logpas_b = self.actor.evaluate(ob_b, ac_b)
            #     ratio = torch.exp(logpas_b - old_logpas_b)
            #
            #     # 限制比率
            #     surr1 = ratio * atarg_b
            #     surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * atarg_b
            #
            #     # 计算policy_loss
            #     policy_loss = -torch.min(surr1, surr2).mean()
            #
            #     # 计算value_loss
            #     vf_loss = F.mse_loss(self.critic(ob_b).squeeze(1), vtarg_b)
            #
            #     # 计算entropy_loss
            #     entropies = self.actor.evaluate_entropy(ob_b)
            #     entropy_loss = -entropies.mean()
            #
            #     # total loss
            #     c1 = self.coef_vf
            #     c2 = self.coef_entpen
            #     loss = policy_loss - c1 * vf_loss + c2 * entropy_loss
            #
            #     # actor - 反向传播优化
            #     policy_loss.backward()
            #     self.actor_optimizer.step()
            #
            #     # critic - 反向传播优化
            #     vf_loss.backward()
            #     # loss.backward()
            #     # self.actor_optimizer.step()
            #     self.critic_optimizer.step()
            #
            #     policy_losses.append(policy_loss.item())
            #     entropy_losses.append(entropy_loss.item())
            #     value_losses.append(vf_loss.item())

    # 模型测试
    def play(self, num_episodes=1,seed=9999):
        # 加载策略
        self.load_weight(self.actor)
        env = gym.make(self.evn_name)
        env.seed(seed)
        scores = []
        for episode in range(num_episodes):
            episode_score = 0
            # 初始化环境
            state = env.reset()

            while True:
                env.render()
                # 选择动作
                with torch.no_grad():
                    action_tensor = self.actor.select_greedy_action(state)
                action = action_tensor.numpy()[0] # single env
                # 执行 action
                next_state, reward, done, _ = env.step(action)
                # 更新 reward
                episode_score += reward
                # 更新 state
                state = next_state

                if done:
                    scores.append(episode_score)
                    break
        avg_score = np.mean(scores)
        print("RESULT: Average score {:.3f} on {} {} games".format(avg_score, num_episodes,self.evn_name))
        env.close()

#初始化
def init_normal_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class Actor(nn.Module):
    def __init__(self, device, input_dim, output_dim,):
        super(Actor, self).__init__()
        self.input_layer = nn.Linear(input_dim, 64)
        self.hidden_layers = nn.ModuleList()
        for idx in range(1):
            self.hidden_layers.append(nn.Linear(64, 64))
        self.output_layer = nn.Linear(64, output_dim)
        self.hfn = torch.tanh
        self.apply(init_normal_weights)
        self.device = device

    def select_action(self, states):
        # 随机采样动作
        probs = self.forward(states)
        dist = Categorical(probs=probs)
        actions = dist.sample()
        # log
        log_probs = dist.log_prob(actions)
        return actions, log_probs

    def select_greedy_action(self, states):
        # 选择可能性最高的动作
        probs = self.forward(states)
        _, actions = probs.max(1)
        return actions

    def get_predictions(self, states, old_actions):
        # 获取旧动作的log_probs和当前分布的熵
        state, old_actions = self._format(states), self._format(old_actions)
        probs = self.forward(states)
        dist = Categorical(probs=probs)
        log_probs = dist.log_prob(old_actions)
        entropies = dist.entropy()
        return log_probs, entropies

    def forward(self, state):
        """给定状态的返回动作概率"""
        state = self._format(state)
        x = self.input_layer(state)
        x = self.hfn(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.hfn(x)
        x = self.output_layer(x)
        x = torch.softmax(x, dim=1)
        return x

    def _format(self, state):
        """将numpy状态转换为张量并添加一个维度"""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
            state = state.unsqueeze(0)
        return state

class Critic(nn.Module):
    def __init__(self, device, input_dim):
        super(Critic, self).__init__()
        self.input_layer = nn.Linear(input_dim, 64)
        self.hidden_layers = nn.ModuleList()
        for idx in range(1):
            self.hidden_layers.append(nn.Linear(64, 64))
        self.output_layer = nn.Linear(64, 1)
        self.hfn = torch.tanh
        self.apply(init_normal_weights)
        self.device = device

    def forward(self, state):
        """返回给定状态的估计值"""
        state = self._format(state)
        x = self.input_layer(state)
        x = self.hfn(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.hfn(x)
        x = self.output_layer(x)
        return x
    def _format(self, state):
        """将numpy状态转换为张量并添加一个维度"""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            state = state.unsqueeze(0)  #
        return state

def main():
    #store_true 是指带触发action时为真，不触发则为假
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--eval",action="store_true")
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    agent = PPO()
    if args.eval:
        """play mode"""
        agent.play(num_episodes=args.eval_episodes, seed=args.seed)
    else:
        print("Training PPO agent on game {}...".format(agent.evn_name))
        agent.train()
        print("Done\n")
        agent.game.close()

if __name__ == "__main__":
    main()
