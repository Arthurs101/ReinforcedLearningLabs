# MonteCarlo.py

from Environment import InventoryEnvironment
import random
from collections import defaultdict

def state_key(state):
    return tuple(sorted(state.items()))

def random_policy(state):
    return {product: random.randint(0, 2) for product in state}

def generate_episode(env, policy, max_days=10):
    episode = []
    state = env.reset()
    for _ in range(max_days):
        action = policy(state)
        next_state, reward = env.step(action)
        episode.append((state.copy(), action.copy(), reward))
        state = next_state
    return episode

def epsilon_greedy_policy(state, Q, epsilon=0.1):
    if random.random() < epsilon:
        return {product: random.randint(0, 2) for product in state}
    else:
        best_action = None
        best_value = float('-inf')
        for a1 in range(3):
            for a2 in range(3):
                action = (a1, a2)
                value = Q[state_key(state)][action]
                if value > best_value:
                    best_value = value
                    best_action = action
        return {'product_A': best_action[0], 'product_B': best_action[1]}

def off_policy_mc_prediction(target_policy, behavior_policy, episodes, gamma=1.0):
    V = defaultdict(float)
    C = defaultdict(float)

    for episode in episodes:
        G = 0
        W = 1.0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            s_key = state_key(state)
            a_key = (action['product_A'], action['product_B'])

            G = gamma * G + reward
            C[s_key] += W
            V[s_key] += (W / C[s_key]) * (G - V[s_key])

            pi = 1.0 if a_key == target_policy(state) else 0.0
            b = 1.0 / 9  # 3x3 posibles acciones
            if b == 0: break
            W *= pi / b
            if W == 0: break
    return V

def exploring_starts_episode(env, policy, max_days=10):
    # Genera estado aleatorio (ya lo hace reset) y aplica una acción aleatoria al inicio
    env.reset()
    first_action = {p: random.randint(0, 2) for p in env.products}
    next_state, reward = env.step(first_action)
    episode = [(env.state.copy(), first_action.copy(), reward)]

    for _ in range(max_days - 1):
        action = policy(env.state)
        next_state, reward = env.step(action)
        episode.append((env.state.copy(), action.copy(), reward))
    return episode

