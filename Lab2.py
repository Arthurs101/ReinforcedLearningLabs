# Simulación de un MDP para un robot en un laberinto 3x3

import random
import numpy as np

# Estados: 3x3 grid -> S = {0,1,2,3,4,5,6,7,8}
# Representación:
# 0 1 2
# 3 4 5
# 6 7 8

# Obstáculos y meta
start_state = 0
goal_state = 8
obstacles = {4}

# Acciones
actions = ['up', 'down', 'left', 'right']

# Transiciones posibles desde cada estado por acción
def next_state(s, a):
    row, col = divmod(s, 3)
    if a == 'up' and row > 0:
        s_ = s - 3
    elif a == 'down' and row < 2:
        s_ = s + 3
    elif a == 'left' and col > 0:
        s_ = s - 1
    elif a == 'right' and col < 2:
        s_ = s + 1
    else:
        s_ = s  # Acción inválida, se queda en el mismo estado
    return s_

# Matriz de transición P[s][a] = {s': prob}
P = {}
R = {}

for s in range(9):
    if s in obstacles:
        continue
    P[s] = {}
    R[s] = {}
    for a in actions:
        s_ = next_state(s, a)
        if s_ in obstacles:
            reward = -10
            s_ = s  # no puede entrar en obstáculo, se queda
        elif s_ == goal_state:
            reward = 10
        else:
            reward = -1  # penalización por moverse
        P[s][a] = {s_: 1.0}  # determinista
        R[s][a] = {s_: reward}

# Política determinista (ejemplo): moverse siempre a la derecha, si no se puede, abajo
policy = {}
for s in range(9):
    if s in obstacles:
        continue
    row, col = divmod(s, 3)
    if col < 2:
        policy[s] = 'right'
    else:
        policy[s] = 'down'

# Simulación de la política
def simulate_policy(start, steps=20):
    state = start
    total_reward = 0
    visited = [state]
    for _ in range(steps):
        if state == goal_state:
            break
        a = policy.get(state, None)
        if a is None:
            break
        transitions = P[state][a]
        next_s = random.choices(list(transitions.keys()), weights=transitions.values())[0]
        reward = R[state][a][next_s]
        total_reward += reward
        state = next_s
        visited.append(state)
    return total_reward, visited

# Evaluar la política varias veces
simulations = 100
results = [simulate_policy(start_state) for _ in range(simulations)]
average_reward = np.mean([r for r, _ in results])

print(f"Recompensa promedio tras {simulations} simulaciones: {average_reward:.2f}")
for i, (reward, path) in enumerate(results[:5]):
    print(f"Simulación {i+1}: Recompensa = {reward}, Camino = {path}")
