import numpy as np
import gymnasium as gym
from typing import Dict, Any, List, Tuple
from envUtils import make_frozenlake_env
from mcts import MCTSAgent
from dynaQAgent import DynaQPlusAgent

def run_mcts(num_episodes: int = 1000,
             seed: int = 777,
             map_name: str = "4x4",
             is_slippery: bool = True,
             num_simulations: int = 200,
             max_depth: int = 30,
             uct_c: float = 1.4,
             gamma: float = 0.99) -> Dict[str, Any]:

    env = make_frozenlake_env(map_name, is_slippery, seed)
    agent = MCTSAgent(env, gamma=gamma, uct_c=uct_c,
                      num_simulations=num_simulations, max_depth=max_depth, seed=seed)

    rewards, successes, steps_list = [], [], []
    visited_pairs = set()

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False
        ep_reward = 0.0
        steps = 0

        while not (done or truncated):
            a = agent.act(int(obs))
            next_obs, r, done, truncated, _ = env.step(a)
            ep_reward += r
            steps += 1
            visited_pairs.add((int(obs), a))
            obs = next_obs
            if done or truncated:
                break

        rewards.append(ep_reward)
        successes.append(1 if ep_reward > 0 else 0)
        steps_list.append(steps)

    env.close()
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    coverage = [len(visited_pairs) / (n_states * n_actions)] * num_episodes

    return {
        "rewards": np.array(rewards),
        "successes": np.array(successes),
        "steps": np.array(steps_list),
        "coverage": np.array(coverage),
    }

def run_dyna_q_plus(num_episodes: int = 1000,
                    seed: int = 777,
                    map_name: str = "4x4",
                    is_slippery: bool = True,
                    alpha: float = 0.1,
                    gamma: float = 0.99,
                    epsilon: float = 0.1,
                    n_planning: int = 20,
                    kappa: float = 1e-3) -> Dict[str, Any]:

    env = make_frozenlake_env(map_name, is_slippery, seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    agent = DynaQPlusAgent(n_states, n_actions, alpha, gamma, epsilon, n_planning, kappa, seed)

    rewards, successes, steps_list = [], [], []
    coverage = []
    visited_pairs = set()

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        truncated = False
        ep_reward = 0.0
        steps = 0

        while not (done or truncated):
            a = agent.choose_action(int(obs))
            next_obs, r, done, truncated, info = env.step(a)
            agent.update(int(obs), a, r, int(next_obs), done or truncated)
            ep_reward += r
            steps += 1
            visited_pairs.add((int(obs), a))
            obs = next_obs
            if done or truncated:
                break

        rewards.append(ep_reward)
        successes.append(1 if ep_reward > 0 else 0)
        steps_list.append(steps)
        coverage.append(len(visited_pairs) / (n_states * n_actions))

    env.close()

    return {
        "rewards": np.array(rewards),
        "successes": np.array(successes),
        "steps": np.array(steps_list),
        "coverage": np.array(coverage),
    }
