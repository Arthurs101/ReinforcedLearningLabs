import os
import csv
import numpy as np
from experiment import run_mcts, run_dyna_q_plus
from plots import plot_success_rate, plot_avg_reward, plot_steps_to_goal, plot_exploration_coverage

def save_csv(metrics: dict, out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'success', 'steps', 'coverage'])
        for i in range(len(metrics['rewards'])):
            writer.writerow([i+1,
                             float(metrics['rewards'][i]),
                             int(metrics['successes'][i]),
                             int(metrics['steps'][i]),
                             float(metrics['coverage'][i])])

def main():
    os.makedirs('outputs', exist_ok=True)

    
    num_episodes = 100

    # MCTS
    mcts_metrics = run_mcts(num_episodes=num_episodes,
                            seed=777,
                            map_name="4x4",
                            is_slippery=True,
                            num_simulations=64,
                            max_depth=20,
                            uct_c=1.4,
                            gamma=0.99)
    save_csv(mcts_metrics, 'outputs/metrics_mcts.csv')

    # Dyna-Q+
    dyna_metrics = run_dyna_q_plus(num_episodes=num_episodes,
                                   seed=777,
                                   map_name="4x4",
                                   is_slippery=True,
                                   alpha=0.1,
                                   gamma=0.99,
                                   epsilon=0.1,
                                   n_planning=10,
                                   kappa=1e-3)
    save_csv(dyna_metrics, 'outputs/metrics_dyna_q_plus.csv')

    # Plots
    plot_success_rate(mcts_metrics['successes'], dyna_metrics['successes'], 'outputs/success_rate.png')
    plot_avg_reward(mcts_metrics['rewards'], dyna_metrics['rewards'], 'outputs/avg_reward.png')
    plot_steps_to_goal(mcts_metrics['steps'], dyna_metrics['steps'], 'outputs/steps_to_goal.png')
    plot_exploration_coverage(mcts_metrics['coverage'], dyna_metrics['coverage'], 'outputs/exploration_coverage.png')

    print("Experimento completado. Revise la carpeta 'outputs/'.")

if __name__ == '__main__':
    main()
