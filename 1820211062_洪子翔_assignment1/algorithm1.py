import argparse
import time
from math import fabs

import gymnasium as gym
import numpy as np
from envs import *

np.set_printoptions(precision=3)

parser = argparse.ArgumentParser(
     description="A program to run assignment 1 implementations.",
     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
     "--env",
     type=str,
     help="The name of the environment to run your algorithm on.",
     choices=["Deterministic-4x4-FrozenLake-v0", "Stochastic-4x4-FrozenLake-v0"],
     default="Deterministic-4x4-FrozenLake-v0",
     # default="Stochastic-4x4-FrozenLake-v0",
)

parser.add_argument(
    "--render-mode",
    "-r",
    type=str,
    help="The render mode for the environment. 'human' opens a window to render. 'ansi' does not render anything.",
    choices=["human", "ansi"],
    default="human",
)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:
    P: nested dictionary of a nested lists
        From gym.core.Environment
        For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a tuple of the form (probability, nextstate, reward, 
        terminal) where    
                - probability: float the probability of transitioning from "state" to "nextstate" with "action"
                - nextstate: int denotes the state we transition to (in range [0, nS - 1])
                - reward: int either 0 or 1, the reward for transitioning from "state" to "nextstate" with "action"
                - terminal: bool True when "nextstate" is a terminal state (hole or goal), False otherwise
     nS: int number of states in the environment
     nA: int number of actions in the environment
     gamma: float Discount factor. Number in range [0, 1)
"""

# 策略评估-固定策略 => 求最优状态值函数
def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """
    Evaluate the value function from a given policy.
    Parameters
    ----------
    P, nS, nA, gamma:
    defined at beginning of file
    policy: np.array[nS]
    The policy to evaluate. Maps states to actions.
    tol: float
    Terminate policy evaluation when max |value_function(s) - prev_value_function(s)| < tol
    Returns
    -------
    value_function: np.ndarray[nS]
    The value function of the given policy, where value_function[s] is the value of state s
    """
    """
        从给定策略评估值函数。
        :param P: 状态转移概率字典，定义了环境动力学
        :param nS: 状态数量
        :param nA: 动作数量
        :param policy: 当前策略
        :param gamma: 折扣因子
        :param tol: 容忍度，用于决定何时停止迭代
        :return: 估计的值函数
        """
    # 初始化v
    value_function = np.zeros(nS)
    # 循环直到收敛
    while True:
        delta = 0
        # 对于每一个状态s进行循环
        for s in range(nS):
            v = value_function[s]
            new_v = 0
            a = policy[s]
            transitions = P[s][a]
            for prob, next_state, reward, terminal in transitions:
                # 使用Bellman方程更新新值
                new_v += prob * (reward + gamma * value_function[next_state])
            # 更新状态s的值函数
            value_function[s] = new_v
            # 计算最大状态值函数的变化，以检查是否收敛
            delta = max(delta, fabs(v - new_v))
        # 如果变化小于容忍度 tol，则停止迭代
        if delta < tol:
            break
    return value_function

# 策略改进-固定状态值函数 => 求最优策略
def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """
    Given the value function from policy improve the policy.
    Parameters
    ----------
    P, nS, nA, gamma: defined at the beginning of the file
    value_from_policy: np.ndarray. The value calculated from the policy
    policy: np.array. The previous policy.
    Returns
    -------
    new_policy: np.ndarray[nS]. An array of integers. Each integer is the optimal action to take
    in that state according to the environment dynamics and the given value function.
    """
    """
        基于值函数改进策略。
        :param P: 状态转移概率字典，定义了环境动力学
        :param nS: 状态数量
        :param nA: 动作数量
        :param value_function: 当前值函数
        :param policy: 当前策略
        :param gamma: 折扣因子
        :return: 新的策略
        """
    # 创建一个新的策略数组
    new_policy = np.zeros(nS, dtype="int")
    # 对每个状态s进行循环
    for s in range(nS):
        # 创建一个数组来存储每个动作的估算值
        q_values = np.zeros(nA)
        # 对每个动作a进行循环
        for a in range(nA):
            # 根据状态 s 和动作 a 查找状态转移概率
            for prob, next_state, reward, terminal in P[s][a]:
                # 使用贝尔曼方程计算状态-动作值函数 Q(s, a)
                q_values[a] += prob * (reward + gamma * value_from_policy[next_state])
            # 选择具有最大估算值的动作作为新策略
        new_policy[s] = np.argmax(q_values)
    return new_policy

# 策略迭代
def policy_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
    Runs policy iteration. You should call the policy_evaluation() and policy_improvement()
    methods to implement this method.
    Parameters
    ----------
    P, nS, nA, gamma: defined at the beginning of the file
    tol: float tol parameter used in policy_evaluation()
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """
    """
        运行策略迭代算法，需要调用policy_evaluation()和policy_improvement()方法。
        :param P: 状态转移概率字典，定义了环境动力学
        :param nS: 状态数量
        :param nA: 动作数量
        :param gamma: 折扣因子
        :param tol: 用于policy_evaluation()中的容忍度
        :return: 值函数和策略
        """
    # 创建策略
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    # 迭代步数
    iters = 0
    while True:
        # 更新步数
        iters += 1
        # 策略评估
        value_function = policy_evaluation(P, nS, nA, policy, gamma, tol)
        # 策略改进
        new_policy = policy_improvement(P, nS, nA, value_function, policy, gamma)
        # 如果策略相同，也就是收敛到一个最优策略
        if np.array_equal(policy, new_policy):
            print(f"policy iteration converges at {iters} rounds")
            return value_function, policy
        # 更新当前策略，继续迭代
        policy = new_policy

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3,max_iter=1000):
    """
    Learn value function and policy by using value iteration method for a given gamma and environment.
    Parameters:
    ----------
    P, nS, nA, gamma: defined at the beginning of the file
    tol: float
    Terminate value iteration when max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """
    """
        使用值迭代方法学习值函数和策略。
        :param P: 状态转移概率字典，定义了环境动力学
        :param nS: 状态数量
        :param nA: 动作数量
        :param gamma: 折扣因子
        :param tol: 容忍度，用于决定何时停止迭代
        :return: 值函数和策略
        """
    # 初始化状态值函数为0
    value_function = np.zeros(nS)
    # 初始化策略为0
    policy = np.zeros(nS, dtype=int)
    # 迭代步数
    iters = 0
    # 迭代多次，最多max_iter次
    for i in range(max_iter):
        delta = 0
        iters += 1
        # 对每个状态 s 进行循环
        for s in range(nS):
            v = value_function[s]
            q_values = np.zeros(nA)
            # 对每个动作 a 进行循环
            for a in range(nA):
                q_val = 0
                transitions = P[s][a]
                # 对每个可能的状态转移进行循环
                for prob, next_state, reward, terminal in transitions:
                    # 使用 Bellman 方程计算动作值
                    q_val += prob * (reward + gamma * value_function[next_state])
                q_values[a] = q_val
            # 选择具有最大动作值的新状态值
            new_v = np.max(q_values)
            # 更新状态 s 的值
            value_function[s] = new_v
            # 计算状态值的变化
            delta = max(delta, np.abs(v - new_v))
        # 如果状态值的变化小于容忍度 tol，则终止迭代
        if delta < tol:
            break

    # 确定基于更新后的值函数的最优策略
    for s in range(nS):
        best_action = np.argmax(
            [sum([p * (r + gamma * value_function[s1]) for p, s1, r, _ in P[s][a]]) for a in range(nA)])
        policy[s] = best_action

    print(f"policy iteration converges at {iters} rounds")
    return value_function, policy

import time  # 导入time模块
def render_single(env, policy, max_steps=100):
    """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!
    Parameters
    ----------
    env: gym.core.Environment to play on. Must have nS, nA, and P as attributes.
    Policy: np.array of shape [env.nS].The action to take at a given state
    """
    episode_reward = 0
    ob, _ = env.reset()
    for t in range(max_steps):
        env.render()
        time.sleep(0.25)  # 添加time.sleep(0.25)
        a = policy[ob]
        ob, rew, done, _, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    env.render()
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)

# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":
    # read in script argument
    args = parser.parse_args()
    # Make gym environment
    env = gym.make(args.env, render_mode=args.render_mode)
    env.nS = env.nrow * env.ncol
    env.nA = 4
    # print(f"Running Policy Iteration for environment {args.env}")
    # print("State Transition Probabilities (P):")
    # for state in range(env.nS):
    #     for action in range(env.nA):
    #         transitions = env.P[state][action]
    #         print(f"State {state}, Action {action}:\n{transitions}")

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)
    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_pi, 100)

    # print(f"Running Value Iteration for environment {args.env}")
    # print("State Transition Probabilities (P):")
    # for state in range(env.nS):
    #     for action in range(env.nA):
    #         transitions = env.P[state][action]
    #         print(f"State {state}, Action {action}:\n{transitions}")

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)
    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3, max_iter=1000)
    render_single(env, p_vi, 100)


