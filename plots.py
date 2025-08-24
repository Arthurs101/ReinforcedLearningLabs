import numpy as np
import matplotlib.pyplot as plt

def plot_success_rate(success_mcts, success_dyna, out_path: str, window: int = 50):
    # Por requerimiento: gráfico individual, sin estilos ni colores específicos.
    mcts_ma = moving_average(success_mcts, window)
    dyna_ma = moving_average(success_dyna, window)

    plt.figure()
    plt.plot(mcts_ma, label='MCTS (MA)')
    plt.plot(dyna_ma, label='Dyna-Q+ (MA)')
    plt.xlabel('Episodio')
    plt.ylabel('Tasa de éxito (MA)')
    plt.title('Tasa de éxito por episodio')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_avg_reward(rew_mcts, rew_dyna, out_path: str, window: int = 50):
    mcts_ma = moving_average(rew_mcts, window)
    dyna_ma = moving_average(rew_dyna, window)

    plt.figure()
    plt.plot(mcts_ma, label='MCTS (MA)')
    plt.plot(dyna_ma, label='Dyna-Q+ (MA)')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa promedio (MA)')
    plt.title('Recompensa promedio por episodio')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_steps_to_goal(steps_mcts, steps_dyna, out_path: str, window: int = 50):
    mcts_ma = moving_average(steps_mcts, window)
    dyna_ma = moving_average(steps_dyna, window)

    plt.figure()
    plt.plot(mcts_ma, label='MCTS (MA)')
    plt.plot(dyna_ma, label='Dyna-Q+ (MA)')
    plt.xlabel('Episodio')
    plt.ylabel('Pasos al objetivo (MA)')
    plt.title('Pasos para alcanzar la meta')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_exploration_coverage(cov_mcts, cov_dyna, out_path: str):
    plt.figure()
    plt.plot(cov_mcts, label='MCTS')
    plt.plot(cov_dyna, label='Dyna-Q+')
    plt.xlabel('Episodio')
    plt.ylabel('Cobertura de (s,a) visitados')
    plt.title('Exploración vs Explotación (Cobertura acumulada)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# Utilidad local (evita dependencia cruzada con env_utils)
def moving_average(data, window: int = 50):
    if len(data) == 0:
        return np.array([])
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='same')
