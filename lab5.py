import numpy as np
import gymnasium as gym
from tqdm import trange
import matplotlib.pyplot as plt
import math
import random
import os
from datetime import datetime

# === Funciones auxiliares (igual que antes) ===
def epsilon_greedy(Q, s, nA, eps):
    if np.random.rand() < eps:
        return np.random.randint(nA)
    else:
        return np.argmax(Q[s])

def expected_value(Q, s, nA, eps):
    greedy_a = np.argmax(Q[s])
    probs = np.ones(nA) * (eps / nA)
    probs[greedy_a] += (1 - eps)
    return np.dot(probs, Q[s])

def rolling_mean(x, w):
    if len(x) < w:
        return np.array(x, dtype=float)
    out = np.convolve(x, np.ones(w)/w, mode="valid")
    pad = np.repeat(out[0], w-1)
    return np.hstack([pad, out])

# === Algoritmos (idénticos a tu código) ===
def run_sarsa(env_id="CliffWalking-v1", episodes=500, alpha=0.5, gamma=1.0,
              eps_start=1.0, eps_end=0.05, eps_decay=0.995, seed=0):
    env = gym.make(env_id)
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    rewards = []
    eps = eps_start
    for ep in range(episodes):
        s, _ = env.reset(seed=seed+ep)
        a = epsilon_greedy(Q, s, nA, eps)
        ep_ret, done = 0, False
        while not done:
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ep_ret += r
            if not done:
                a2 = epsilon_greedy(Q, s2, nA, eps)
                td_target = r + gamma * Q[s2, a2]
            else:
                td_target = r
            Q[s, a] += alpha * (td_target - Q[s, a])
            s, a = s2, (epsilon_greedy(Q, s2, nA, eps) if not done else 0)
        rewards.append(ep_ret)
        eps = max(eps_end, eps * eps_decay)
    env.close()
    return Q, rewards

def run_expected_sarsa(env_id="CliffWalking-v1", episodes=500, alpha=0.5, gamma=1.0,
                       eps_start=1.0, eps_end=0.05, eps_decay=0.995, seed=1):
    env = gym.make(env_id)
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    rewards = []
    eps = eps_start
    for ep in range(episodes):
        s, _ = env.reset(seed=seed+ep)
        ep_ret, done = 0, False
        while not done:
            a = epsilon_greedy(Q, s, nA, eps)
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ep_ret += r
            if not done:
                exp_q = expected_value(Q, s2, nA, eps)
                td_target = r + gamma * exp_q
            else:
                td_target = r
            Q[s, a] += alpha * (td_target - Q[s, a])
            s = s2
        rewards.append(ep_ret)
        eps = max(eps_end, eps * eps_decay)
    env.close()
    return Q, rewards

def run_q_learning(env_id="CliffWalking-v1", episodes=500, alpha=0.5, gamma=1.0,
                   eps_start=1.0, eps_end=0.05, eps_decay=0.995, seed=2):
    env = gym.make(env_id)
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    rewards = []
    eps = eps_start
    for ep in range(episodes):
        s, _ = env.reset(seed=seed+ep)
        ep_ret, done = 0, False
        while not done:
            a = epsilon_greedy(Q, s, nA, eps)
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ep_ret += r
            td_target = r if done else r + gamma * np.max(Q[s2])
            Q[s, a] += alpha * (td_target - Q[s, a])
            s = s2
        rewards.append(ep_ret)
        eps = max(eps_end, eps * eps_decay)
    env.close()
    return Q, rewards

def run_n_step_sarsa(env_id="CliffWalking-v1", episodes=500, n=4, alpha=0.5, gamma=1.0,
                     eps_start=1.0, eps_end=0.05, eps_decay=0.995, seed=3):
    env = gym.make(env_id)
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    rewards_per_ep = []
    eps = eps_start
    for ep in range(episodes):
        S, A, R = [], [], [0.0]
        s, _ = env.reset(seed=seed+ep)
        a = epsilon_greedy(Q, s, nA, eps)
        S.append(s)
        A.append(a)
        T = math.inf
        t = 0
        ep_ret = 0
        while True:
            if t < T:
                s2, r, term, trunc, _ = env.step(A[t])
                done = term or trunc
                ep_ret += r
                S.append(s2)
                R.append(r)
                if done:
                    T = t + 1
                else:
                    a2 = epsilon_greedy(Q, s2, nA, eps)
                    A.append(a2)
            tau = t - n + 1
            if tau >= 0:
                G = 0.0
                upper = min(tau + n, T)
                for k in range(tau + 1, upper + 1):
                    G += (gamma ** (k - tau - 1)) * R[k]
                if tau + n < T:
                    G += (gamma ** n) * Q[S[tau + n], A[tau + n]]
                Q[S[tau], A[tau]] += alpha * (G - Q[S[tau], A[tau]])
            if tau == T - 1:
                break
            t += 1
        rewards_per_ep.append(ep_ret)
        eps = max(eps_end, eps * eps_decay)
    env.close()
    return Q, rewards_per_ep

# === Experimento y guardado de resultados ===
def run_experiment(env_id="CliffWalking-v1", episodes=800, runs=5, alpha=0.5, gamma=1.0,
                   n=4, eps_start=1.0, eps_end=0.05, eps_decay=0.995, smooth_window=25):
    algos = {
        "SARSA": run_sarsa,
        "ExpectedSARSA": run_expected_sarsa,
        f"nStepSARSA(n={n})": lambda **kw: run_n_step_sarsa(n=n, **kw),
        "Qlearning": run_q_learning,
    }
    history = {}
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(results_dir, f"results_{timestamp}.txt")

    with open(txt_path, "w") as f:
        f.write(f"Resultados CliffWalking-v1 ({timestamp})\n\n")
        for i, (name, fn) in enumerate(algos.items()):
            all_rewards = []
            for rseed in range(runs):
                _, rewards = fn(env_id=env_id, episodes=episodes, alpha=alpha, gamma=gamma,
                                eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay,
                                seed=1000*i + rseed)
                all_rewards.append(rewards)
            mean_rewards = np.mean(np.stack(all_rewards, axis=0), axis=0)
            history[name] = rolling_mean(mean_rewards, smooth_window)

            # Retorno final
            final_return = mean_rewards[-1]
            avg_total = np.mean(mean_rewards)

            # Promedios cada 100 episodios
            block_size = 100
            block_avgs = [
                np.mean(mean_rewards[i:i+block_size])
                for i in range(0, len(mean_rewards), block_size)
            ]

            # Guardar en TXT
            f.write(f"{name}:\n")
            f.write(f"  Retorno medio final: {final_return:.2f}\n")
            f.write(f"  Retorno medio total: {avg_total:.2f}\n")
            f.write("  Promedios cada 100 episodios:\n")
            for idx, avg in enumerate(block_avgs, start=1):
                start_ep = (idx-1)*block_size + 1
                end_ep = min(idx*block_size, episodes)
                f.write(f"    Episodios {start_ep}-{end_ep}: {avg:.2f}\n")
            f.write("\n")


    # Guardar gráfico
    plt.figure(figsize=(9, 5))
    for name, curve in history.items():
        plt.plot(curve, label=name)
    plt.xlabel("Episodio")
    plt.ylabel(f"Retorno (promedio suavizado w={smooth_window})")
    plt.title("CliffWalking-v1 — Comparación de Algoritmos")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"plot_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Resultados guardados en: {txt_path}")
    print(f"Gráfico guardado en: {plot_path}")

if __name__ == "__main__":
    run_experiment()
