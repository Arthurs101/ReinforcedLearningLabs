"""Command-line utility to train or evaluate agents on the SUMO environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict

import numpy as np

from agents import AgentTransition, DQNAgent, DQNConfig
from sumo import SUMOEnvironment, SUMOEnvironmentConfig, OSMScenario, SimpleIntersectionScenario


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
    # OSM file support
    parser.add_argument("--osm-file", type=Path, default=None, help="Path to OSM map file (if provided, uses OSM scenario instead of simple intersection)")
    parser.add_argument("--flow-rate", type=int, default=600, help="Vehicle flow rate per hour (for OSM scenarios)")
    parser.add_argument("--tls-id", type=str, default=None, help="Traffic light ID to control (for OSM scenarios, defaults to first TLS found)")
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


def run_episode(env: SUMOEnvironment, agent: DQNAgent, train: bool = True) -> Tuple[float, dict]:
    observation, _ = env.reset()
    done = False
    episode_reward = 0.0
    episode_info = None

    while not done:
        action = agent.select_action(observation, explore=train)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_info = info  # Keep last info for final metrics

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

    return episode_reward, episode_info or {}


def evaluate(env: SUMOEnvironment, agent: DQNAgent, episodes: int) -> Tuple[float, dict]:
    results = [run_episode(env, agent, train=False) for _ in range(episodes)]
    rewards = [r[0] for r in results]
    infos = [r[1] for r in results]
    
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    avg_queues = float(np.mean([sum(info.get("queues", [])) for info in infos])) if infos else 0.0
    avg_waiting = float(np.mean([sum(info.get("waiting_times", [])) for info in infos])) if infos else 0.0
    
    metrics = {
        "avg_reward": avg_reward,
        "avg_total_queues": avg_queues,
        "avg_total_waiting_time": avg_waiting,
    }
    return avg_reward, metrics


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    env_config, agent_config = build_configs(args)

    # Create scenario based on whether OSM file is provided
    if args.osm_file:
        if not args.osm_file.exists():
            print(f"Error: OSM file not found: {args.osm_file}", file=sys.stderr)
            return 1
        # Calculate end_time based on max_steps and step_length
        end_time = int(args.max_steps * args.step_length)
        scenario = OSMScenario(
            osm_file=args.osm_file,
            flow_rate=args.flow_rate,
            tls_id=args.tls_id,
            begin_time=0,
            end_time=end_time,
        )
        print(f"Using OSM scenario from: {args.osm_file}")
    else:
        scenario = SimpleIntersectionScenario()
        print("Using simple intersection scenario")

    env = SUMOEnvironment(config=env_config, scenario=scenario)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, agent_config)
    
    if args.osm_file:
        print(f"Traffic light ID: {env.artifacts.tls_id}")
        print(f"Incoming lanes: {len(env.artifacts.incoming_lanes)}")
        print(f"Outgoing lanes: {len(env.artifacts.outgoing_lanes)}")

    if args.no_train:
        score, metrics = evaluate(env, agent, args.eval_episodes)
        print(f"\nEvaluation Results (untrained agent):")
        print(f"  Average Reward: {score:.2f}")
        print(f"  Average Total Queues: {metrics['avg_total_queues']:.1f}")
        print(f"  Average Total Waiting Time: {metrics['avg_total_waiting_time']:.2f}s")
        env.close()
        return 0

    best_eval = -float("inf")
    print("\nStarting training...")
    print("=" * 80)
    
    for episode in range(1, args.episodes + 1):
        reward, info = run_episode(env, agent, train=True)
        total_queues = sum(info.get("queues", []))
        total_waiting = sum(info.get("waiting_times", []))
        
        print(f"Episode {episode:03d} | Reward: {reward:7.2f} | "
              f"Epsilon: {agent.epsilon:.3f} | "
              f"Total Queues: {total_queues:3d} | "
              f"Total Waiting: {total_waiting:6.2f}s")

        if args.eval_every and episode % args.eval_every == 0:
            eval_reward, eval_metrics = evaluate(env, agent, args.eval_episodes)
            print(f"\n  Evaluation (episodes {episode - args.eval_every + 1}-{episode}):")
            print(f"    Average Reward: {eval_reward:.2f}")
            print(f"    Average Total Queues: {eval_metrics['avg_total_queues']:.1f}")
            print(f"    Average Total Waiting Time: {eval_metrics['avg_total_waiting_time']:.2f}s")
            best_eval = max(best_eval, eval_reward)
            print()

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.save_path)
    print(f"Saved trained agent to {args.save_path}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

