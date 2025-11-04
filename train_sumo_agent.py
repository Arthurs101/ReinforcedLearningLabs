"""Command-line utility to train or evaluate agents on the SUMO environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

from agents import AgentTransition, DQNAgent, DQNConfig
from sumo import SUMOEnvironment, SUMOEnvironmentConfig


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent on the SUMO intersection environment")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=900, help="Maximum steps per episode")
    parser.add_argument("--use-gui", action="store_true", help="Launch SUMO with GUI rendering")
    parser.add_argument("--step-length", type=float, default=1.0, help="Simulation step length")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for SUMO")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device for PyTorch (cpu/cuda)")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for DQN agent")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for replay updates")
    parser.add_argument("--replay-size", type=int, default=50_000, help="Replay buffer capacity")
    parser.add_argument("--target-update", type=int, default=200, help="Steps between target network updates")
    parser.add_argument("--eval-every", type=int, default=0, help="Evaluate every N episodes (0 disables)")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes to average during evaluation runs")
    parser.add_argument("--save-path", type=Path, default=Path("artifacts/dqn_agent.pt"), help="Where to save the trained agent")
    parser.add_argument("--no-train", action="store_true", help="Skip learning and run evaluation only")
    return parser.parse_args(argv)


def build_configs(args: argparse.Namespace) -> Tuple[SUMOEnvironmentConfig, DQNConfig]:
    env_config = SUMOEnvironmentConfig(
        max_steps=args.max_steps,
        step_length=args.step_length,
        use_gui=args.use_gui,
        seed=args.seed,
    )

    agent_config = DQNConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        target_update_interval=args.target_update,
        device=args.device,
    )

    return env_config, agent_config


def run_episode(env: SUMOEnvironment, agent: DQNAgent, train: bool = True) -> float:
    observation, _ = env.reset()
    done = False
    episode_reward = 0.0

    while not done:
        action = agent.select_action(observation, explore=train)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if train:
            transition = AgentTransition(
                state=observation,
                action=action,
                reward=reward,
                next_state=next_obs,
                done=done,
            )
            agent.observe(transition)
            agent.update()

        observation = next_obs
        episode_reward += reward

    return episode_reward


def evaluate(env: SUMOEnvironment, agent: DQNAgent, episodes: int) -> float:
    rewards = [run_episode(env, agent, train=False) for _ in range(episodes)]
    return float(np.mean(rewards)) if rewards else 0.0


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    env_config, agent_config = build_configs(args)

    env = SUMOEnvironment(env_config)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, agent_config)

    if args.no_train:
        score = evaluate(env, agent, args.eval_episodes)
        print(f"Evaluation reward (untrained agent): {score:.2f}")
        env.close()
        return 0

    best_eval = -float("inf")
    for episode in range(1, args.episodes + 1):
        reward = run_episode(env, agent, train=True)
        print(f"Episode {episode:03d} | reward={reward:.2f} | epsilon={agent.epsilon:.3f}")

        if args.eval_every and episode % args.eval_every == 0:
            eval_reward = evaluate(env, agent, args.eval_episodes)
            print(f"  Evaluation mean reward over {args.eval_episodes} episodes: {eval_reward:.2f}")
            best_eval = max(best_eval, eval_reward)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.save_path)
    print(f"Saved trained agent to {args.save_path}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

