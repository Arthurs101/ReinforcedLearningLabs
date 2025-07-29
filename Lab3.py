import random
import numpy as np

# Parámetros del MDP
start_state = 0
goal_state = 8
obstacles = {4}
actions = ['up', 'down', 'left', 'right']
gamma = 0.9
threshold = 1e-3

# Función para determinar el siguiente estado dado un estado y una acción
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
        s_ = s
    return s_

# Construir matriz de transición P y función de recompensa R
P, R = {}, {}
for s in range(9):
    if s in obstacles:
        continue
    P[s] = {}
    R[s] = {}
    for a in actions:
        s_ = next_state(s, a)
        if s_ in obstacles:
            reward = -10
            s_ = s
        elif s_ == goal_state:
            reward = 10
        else:
            reward = -1
        P[s][a] = {s_: 1.0}
        R[s][a] = {s_: reward}

# Inicializar función de valor
V = {s: 0 for s in range(9) if s not in obstacles}

# Iteración de Valor
def value_iteration():
    V = {s: 0 for s in P}
    while True:
        delta = 0
        for s in P:
            v = V[s]
            action_values = []
            for a in actions:
                value = 0
                for s_, prob in P[s][a].items():
                    value += prob * (R[s][a][s_] + gamma * V[s_])
                action_values.append(value)
            V[s] = max(action_values)
            delta = max(delta, abs(v - V[s]))
        if delta < threshold:
            break
    # Extraer política óptima
    policy = {}
    for s in P:
        best_a = None
        best_val = float('-inf')
        for a in actions:
            value = sum(prob * (R[s][a][s_] + gamma * V[s_]) for s_, prob in P[s][a].items())
            if value > best_val:
                best_val = value
                best_a = a
        policy[s] = best_a
    return V, policy

# Iteración de Póliza
def policy_iteration():
    policy = {s: random.choice(actions) for s in P}
    V = {s: 0 for s in P}
    stable = False

    while not stable:
        # Evaluación de la política
        while True:
            delta = 0
            for s in P:
                v = V[s]
                a = policy[s]
                V[s] = sum(prob * (R[s][a][s_] + gamma * V[s_]) for s_, prob in P[s][a].items())
                delta = max(delta, abs(v - V[s]))
            if delta < threshold:
                break

        # Mejora de la política
        stable = True
        for s in P:
            old_action = policy[s]
            best_action = max(actions, key=lambda a: sum(
                prob * (R[s][a][s_] + gamma * V[s_]) for s_, prob in P[s][a].items()))
            if old_action != best_action:
                policy[s] = best_action
                stable = False
    return V, policy

# Ejecutar ambos algoritmos
V_vi, pi_vi = value_iteration()
V_pi, pi_pi = policy_iteration()

# Mostrar resultados
print("\nFunción de valor óptima (Iteración de Valor):")
for s in range(9):
    if s in V_vi:
        print(f"V[{s}] = {V_vi[s]:.3f}")
print("\nPolítica óptima (Iteración de Valor):")
for s in sorted(pi_vi):
    print(f"π[{s}] = {pi_vi[s]}")

print("\nFunción de valor óptima (Iteración de Póliza):")
for s in range(9):
    if s in V_pi:
        print(f"V[{s}] = {V_pi[s]:.3f}")
print("\nPolítica óptima (Iteración de Póliza):")
for s in sorted(pi_pi):
    print(f"π[{s}] = {pi_pi[s]}")

"""
Resultados de ejeccuión:
    
(rl_env) buntu@DESKTOP-QSIFHMU:~/Uni/2nd2025/ReinforcedLearning$ /home/buntu/Uni/2nd2025/ReinforcedLearning/rl_env/bin/python /home/buntu/Uni/2nd2025/ReinforcedLearning/Lab3.py

Función de valor óptima (Iteración de Valor):
V[0] = 70.182
V[1] = 79.092
V[2] = 88.992
V[3] = 79.092
V[5] = 99.992
V[6] = 88.992
V[7] = 99.992
V[8] = 99.992

Política óptima (Iteración de Valor):
π[0] = down
π[1] = right
π[2] = down
π[3] = down
π[5] = down
π[6] = right
π[7] = right
π[8] = down

Función de valor óptima (Iteración de Póliza):
V[0] = 70.186
V[1] = 79.096
V[2] = 88.996
V[3] = 79.096
V[5] = 99.996
V[6] = 88.996
V[7] = 99.996
V[8] = 99.996

Política óptima (Iteración de Póliza):
π[0] = down
π[1] = right
π[2] = down
π[3] = down
π[5] = down
π[6] = right
π[7] = right
π[8] = down
"""