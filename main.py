from collections import defaultdict
from Environment import InventoryEnvironment
from MonteCarlo import (
    generate_episode,
    exploring_starts_episode,
    epsilon_greedy_policy,
    off_policy_mc_prediction
)
import random
import numpy as np

env = InventoryEnvironment()
Q = defaultdict(lambda: defaultdict(float))

output_lines = []

output_lines.append("=== 1. Evaluación con política epsilon-greedy (epsilon=0.1) ===")
episodes = [generate_episode(env, lambda s: epsilon_greedy_policy(s, Q, 0.1)) for _ in range(1000)]
rewards = [sum(r for _, _, r in ep) for ep in episodes]
output_lines.append(f"Recompensa promedio: {np.mean(rewards):.2f}\n")

output_lines.append("=== 2. Evaluación con Exploring Starts ===")
episodes_es = [exploring_starts_episode(env, lambda s: epsilon_greedy_policy(s, Q, 0.1)) for _ in range(1000)]
rewards_es = [sum(r for _, _, r in ep) for ep in episodes_es]
output_lines.append(f"Recompensa promedio (Exploring Starts): {np.mean(rewards_es):.2f}\n")

output_lines.append("=== 3. Comparación de distintos valores de epsilon ===")
epsilons = [0.0, 0.1, 0.2, 0.5]
for eps in epsilons:
    rwd = []
    for _ in range(200):
        ep = generate_episode(env, lambda s: epsilon_greedy_policy(s, Q, eps))
        rwd.append(sum(r for _, _, r in ep))
    output_lines.append(f"Epsilon {eps:.2f} -> Recompensa promedio: {np.mean(rwd):.2f}")
output_lines.append("")

output_lines.append("=== 4. Evaluación con aprendizaje Off-Policy ===")
def target_policy(state):
    return (1, 1)

def behavior_policy(state):
    return {p: random.randint(0, 2) for p in state}

behavior_episodes = [generate_episode(env, behavior_policy) for _ in range(1000)]
V = off_policy_mc_prediction(lambda s: target_policy(s), behavior_policy, behavior_episodes)

output_lines.append("Valores estimados V(s) usando off-policy:")
for state, value in V.items():
    output_lines.append(f"{state}: {value:.2f}")

# Save results to a TXT file
with open("results.txt", "w") as f:
    for line in output_lines:
        print(line)
        f.write(line + "\n")
