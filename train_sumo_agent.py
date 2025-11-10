"""Command-line utility to train or evaluate agents on the SUMO environment."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Union

import numpy as np

from agents import (
    AgentTransition,
    DQNAgent,
    DQNConfig,
    ImprovedDQNAgent,
    ImprovedDQNConfig,
)
from sumo import SUMOEnvironment, SUMOEnvironmentConfig, OSMScenario, SimpleIntersectionScenario
import gymnasium as gym


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
    parser.add_argument("--save-path", type=Path, default=None, help="Where to save the trained agent (default: output/run_id/weights.pt)")
    parser.add_argument("--no-train", action="store_true", help="Skip learning and run evaluation only")
    parser.add_argument("--eval-gui", action="store_true", help="Run evaluation with GUI visualization (requires --load-path)")
    parser.add_argument("--load-path", type=Path, default=None, help="Path to load trained agent weights for evaluation")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory to save run outputs (default: output)")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID for this training session (default: auto-generated)")
    # OSM file support
    parser.add_argument("--osm-file", type=Path, default=None, help="Path to OSM map file (if provided, uses OSM scenario instead of simple intersection)")
    parser.add_argument("--flow-rate", type=int, default=600, help="Vehicle flow rate per hour (for OSM scenarios)")
    parser.add_argument("--tls-id", type=str, default=None, help="Traffic light ID to control (for OSM scenarios, defaults to first TLS found)")
    parser.add_argument("--agent-type", type=str, default="dqn", choices=["dqn", "improved"], help="Agent type: 'dqn' (standard) or 'improved' (Double DQN + Dueling)")
    return parser.parse_args(argv)


def build_configs(args: argparse.Namespace, agent_type: str = "dqn") -> Tuple[SUMOEnvironmentConfig, Union[DQNConfig, ImprovedDQNConfig]]:
    """Build environment and agent configurations."""
    env_config = SUMOEnvironmentConfig(
        max_steps=args.max_steps,
        step_length=args.step_length,
        use_gui=args.use_gui,
        seed=args.seed,
    )

    if agent_type == "improved":
        agent_config = ImprovedDQNConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            replay_size=args.replay_size,
            target_update_interval=args.target_update,
            device=args.device,
        )
    else:
        agent_config = DQNConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            replay_size=args.replay_size,
            target_update_interval=args.target_update,
            device=args.device,
        )

    return env_config, agent_config


def run_episode(env: SUMOEnvironment, agent: Union[DQNAgent, ImprovedDQNAgent], train: bool = True) -> Tuple[float, dict]:
    observation, _ = env.reset()
    done = False
    episode_reward = 0.0
    episode_info = None
    episode_losses = []
    episode_actions = []

    while not done:
        action = agent.select_action(observation, explore=train)
        
        # Convert flattened action back to tuple/array for MultiDiscrete
        if isinstance(env.action_space, gym.spaces.MultiDiscrete):
            num_durations = env.action_space.nvec[1]
            phase_idx = action // num_durations
            duration_idx = action % num_durations
            env_action = np.array([phase_idx, duration_idx])
        else:
            env_action = action
        
        next_obs, reward, terminated, truncated, info = env.step(env_action)
        done = terminated or truncated
        episode_info = info  # Keep last info for final metrics
        episode_actions.append(action)  # Store flattened action for compatibility

        if train:
            transition = AgentTransition(
                state=observation,
                action=action,  # Store flattened action
                reward=reward,
                next_state=next_obs,
                done=done,
            )
            agent.observe(transition)
            update_metrics = agent.update()
            if update_metrics and "loss" in update_metrics:
                episode_losses.append(update_metrics["loss"])

        observation = next_obs
        episode_reward += reward

    # Calculate action distribution
    action_counts = {}
    for action in episode_actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    episode_info = episode_info or {}
    episode_info["losses"] = episode_losses
    episode_info["actions"] = episode_actions
    episode_info["action_distribution"] = action_counts
    episode_info["avg_loss"] = float(np.mean(episode_losses)) if episode_losses else None
    episode_info["num_updates"] = len(episode_losses)

    return episode_reward, episode_info


def evaluate(env: SUMOEnvironment, agent: Union[DQNAgent, ImprovedDQNAgent], episodes: int) -> Tuple[float, dict]:
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


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{unique_id}"


def save_metrics(run_dir: Path, metrics: Dict) -> None:
    """Save metrics to JSON file."""
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")


def save_config(run_dir: Path, args: argparse.Namespace, run_id: str) -> None:
    """Save training configuration for reproducibility."""
    config = {
        "run_id": run_id,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "step_length": args.step_length,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "replay_size": args.replay_size,
        "target_update": args.target_update,
        "eval_every": args.eval_every,
        "eval_episodes": args.eval_episodes,
        "osm_file": str(args.osm_file) if args.osm_file else None,
        "flow_rate": args.flow_rate,
        "tls_id": args.tls_id,
        "device": args.device,
    }
    config_file = run_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_file}")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    
    # Generate or use provided run ID
    run_id = args.run_id or generate_run_id()
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Run ID: {run_id}")
    print(f"Output directory: {run_dir}")
    
    # Save configuration
    save_config(run_dir, args, run_id)

    env_config, agent_config = build_configs(args, args.agent_type)

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
    
    # Handle MultiDiscrete action space (phase + duration) vs Discrete (phase only)
    if isinstance(env.action_space, gym.spaces.MultiDiscrete):
        # Flatten MultiDiscrete: [num_phases, num_durations] -> num_phases * num_durations
        action_size = env.action_space.nvec[0] * env.action_space.nvec[1]
        print(f"Action space: MultiDiscrete [{env.action_space.nvec[0]} phases, {env.action_space.nvec[1]} durations] = {action_size} total actions")
    else:
        action_size = env.action_space.n
        print(f"Action space: Discrete ({action_size} phases)")
    
    # Create agent based on type
    if args.agent_type == "improved":
        agent = ImprovedDQNAgent(state_size, action_size, agent_config)
        print(f"Using Improved DQN agent (Double DQN + Dueling architecture)")
    else:
        agent = DQNAgent(state_size, action_size, agent_config)
        print(f"Using standard DQN agent")
    
    if args.osm_file:
        print(f"Traffic light ID: {env.artifacts.tls_id}")
        print(f"Incoming lanes: {len(env.artifacts.incoming_lanes)}")
        print(f"Outgoing lanes: {len(env.artifacts.outgoing_lanes)}")

    # Handle GUI evaluation mode
    if args.eval_gui:
        if not args.load_path or not args.load_path.exists():
            print(f"Error: --load-path is required for --eval-gui mode", file=sys.stderr)
            return 1
        agent.load(args.load_path)
        print(f"Loaded agent from {args.load_path}")
        
        # Create GUI config for evaluation
        eval_config = SUMOEnvironmentConfig(
            max_steps=args.max_steps,
            step_length=args.step_length,
            use_gui=True,  # Force GUI for visualization
            seed=args.seed,
        )
        eval_env = SUMOEnvironment(config=eval_config, scenario=scenario)
        
        print("\nRunning evaluation with GUI...")
        print(f"Running {args.eval_episodes} episodes. Watch the SUMO GUI window.")
        score, metrics = evaluate(eval_env, agent, args.eval_episodes)
        
        print(f"\nEvaluation Results:")
        print(f"  Average Reward: {score:.2f}")
        print(f"  Average Total Queues: {metrics['avg_total_queues']:.1f}")
        print(f"  Average Total Waiting Time: {metrics['avg_total_waiting_time']:.2f}s")
        
        # Save evaluation metrics
        eval_metrics = {
            "run_id": run_id,
            "mode": "evaluation_gui",
            "episodes": args.eval_episodes,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        save_metrics(run_dir, eval_metrics)
        
        eval_env.close()
        env.close()
        return 0

    if args.no_train:
        if args.load_path and args.load_path.exists():
            agent.load(args.load_path)
            print(f"Loaded agent from {args.load_path}")
        
        score, metrics = evaluate(env, agent, args.eval_episodes)
        print(f"\nEvaluation Results:")
        print(f"  Average Reward: {score:.2f}")
        print(f"  Average Total Queues: {metrics['avg_total_queues']:.1f}")
        print(f"  Average Total Waiting Time: {metrics['avg_total_waiting_time']:.2f}s")
        
        # Save evaluation metrics
        eval_metrics = {
            "run_id": run_id,
            "mode": "evaluation",
            "episodes": args.eval_episodes,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        save_metrics(run_dir, eval_metrics)
        
        env.close()
        return 0

    # Training mode
    best_eval = -float("inf")
    training_metrics = {
        "run_id": run_id,
        "mode": "training",
        "episodes": args.episodes,
        "episode_data": [],
        "evaluation_data": [],
        "config": {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "replay_size": args.replay_size,
            "target_update": args.target_update,
            "flow_rate": args.flow_rate,
            "osm_file": str(args.osm_file) if args.osm_file else None,
            "tls_id": args.tls_id,
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    print("\nStarting training...")
    print("=" * 80)
    
    for episode in range(1, args.episodes + 1):
        reward, info = run_episode(env, agent, train=True)
        total_queues = sum(info.get("queues", []))
        total_waiting = sum(info.get("waiting_times", []))
        
        episode_data = {
            "episode": episode,
            "reward": float(reward),
            "epsilon": float(agent.epsilon),
            "total_queues": int(total_queues),
            "total_waiting_time": float(total_waiting),
            "queues": info.get("queues", []),
            "waiting_times": info.get("waiting_times", []),
            "avg_loss": info.get("avg_loss"),
            "num_updates": info.get("num_updates", 0),
            "action_distribution": info.get("action_distribution", {}),
        }
        training_metrics["episode_data"].append(episode_data)
        
        print(f"Episode {episode:03d} | Reward: {reward:7.2f} | "
              f"Epsilon: {agent.epsilon:.3f} | "
              f"Total Queues: {total_queues:3d} | "
              f"Total Waiting: {total_waiting:6.2f}s")

        if args.eval_every and episode % args.eval_every == 0:
            eval_reward, eval_metrics = evaluate(env, agent, args.eval_episodes)
            eval_data = {
                "episode": episode,
                "metrics": eval_metrics,
            }
            training_metrics["evaluation_data"].append(eval_data)
            
            print(f"\n  Evaluation (episodes {episode - args.eval_every + 1}-{episode}):")
            print(f"    Average Reward: {eval_reward:.2f}")
            print(f"    Average Total Queues: {eval_metrics['avg_total_queues']:.1f}")
            print(f"    Average Total Waiting Time: {eval_metrics['avg_total_waiting_time']:.2f}s")
            best_eval = max(best_eval, eval_reward)
            print()

    # Save agent weights
    weights_path = args.save_path or (run_dir / "weights.pt")
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(weights_path)
    print(f"\nSaved trained agent weights to {weights_path}")
    
    # Add final statistics to metrics
    training_metrics["final_epsilon"] = float(agent.epsilon)
    training_metrics["best_eval_reward"] = float(best_eval) if best_eval > -float("inf") else None
    
    # Save metrics
    save_metrics(run_dir, training_metrics)
    
    print(f"\nTraining complete! Run ID: {run_id}")
    print(f"  Weights: {weights_path}")
    print(f"  Metrics: {run_dir / 'metrics.json'}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

