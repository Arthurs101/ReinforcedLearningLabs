"""Visualize a trained agent controlling traffic lights in SUMO GUI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from agents import DQNAgent, DQNConfig, ImprovedDQNAgent, ImprovedDQNConfig
from sumo import SUMOEnvironment, SUMOEnvironmentConfig, OSMScenario, SimpleIntersectionScenario
import gymnasium as gym
import numpy as np


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a trained DQN agent controlling traffic lights in SUMO GUI"
    )
    parser.add_argument(
        "--load-path",
        type=Path,
        required=True,
        help="Path to trained agent weights file (e.g., output/run_id/weights.pt)",
    )
    parser.add_argument(
        "--osm-file",
        type=Path,
        default=None,
        help="Path to OSM map file (required if model was trained on OSM scenario)",
    )
    parser.add_argument(
        "--flow-rate",
        type=int,
        default=600,
        help="Vehicle flow rate per hour (for OSM scenarios, should match training)",
    )
    parser.add_argument(
        "--tls-id",
        type=str,
        default=None,
        help="Traffic light ID to control (for OSM scenarios, defaults to first TLS found)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run (default: 5)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=900,
        help="Maximum steps per episode (default: 900)",
    )
    parser.add_argument(
        "--step-length",
        type=float,
        default=1.0,
        help="Simulation step length in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for SUMO (default: None)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without GUI (headless mode, faster but no visualization)",
    )
    return parser.parse_args(argv)


def load_config_from_checkpoint(weights_path: Path) -> Optional[dict]:
    """Try to load configuration from the checkpoint directory."""
    config_path = weights_path.parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    # Check if weights file exists
    if not args.load_path.exists():
        print(f"Error: Weights file not found: {args.load_path}", file=sys.stderr)
        return 1

    # Try to load config from checkpoint directory
    checkpoint_config = load_config_from_checkpoint(args.load_path)
    if checkpoint_config:
        print(f"Found checkpoint config from: {args.load_path.parent / 'config.json'}")
        # Use config values if not explicitly provided
        if args.osm_file is None and checkpoint_config.get("osm_file"):
            args.osm_file = Path(checkpoint_config["osm_file"])
            print(f"Using OSM file from checkpoint config: {args.osm_file}")
        if args.flow_rate == 600 and checkpoint_config.get("flow_rate"):
            args.flow_rate = checkpoint_config["flow_rate"]
            print(f"Using flow rate from checkpoint config: {args.flow_rate}")
        if args.max_steps == 900 and checkpoint_config.get("max_steps"):
            args.max_steps = checkpoint_config["max_steps"]
        if args.step_length == 1.0 and checkpoint_config.get("step_length"):
            args.step_length = checkpoint_config["step_length"]

    # Create environment configuration
    env_config = SUMOEnvironmentConfig(
        max_steps=args.max_steps,
        step_length=args.step_length,
        use_gui=not args.no_gui,
        seed=args.seed,
    )

    # Create scenario
    if args.osm_file:
        if not args.osm_file.exists():
            print(f"Error: OSM file not found: {args.osm_file}", file=sys.stderr)
            return 1
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

    # Create environment
    env = SUMOEnvironment(config=env_config, scenario=scenario)
    state_size = env.observation_space.shape[0]
    
    # Handle MultiDiscrete action space
    if isinstance(env.action_space, gym.spaces.MultiDiscrete):
        action_size = env.action_space.nvec[0] * env.action_space.nvec[1]
    else:
        action_size = env.action_space.n

    # Try to detect agent type from checkpoint
    agent = None
    try:
        import torch
        checkpoint = torch.load(args.load_path, map_location="cpu", weights_only=False)
        
        # Check if checkpoint has config that indicates improved agent
        config = checkpoint.get("config")
        if config and hasattr(config, "use_double_dqn"):
            # Improved agent
            agent_config = ImprovedDQNConfig(device="cpu")
            agent = ImprovedDQNAgent(state_size, action_size, agent_config)
            print("Detected Improved DQN agent from checkpoint")
        else:
            # Standard DQN agent
            agent_config = DQNConfig(device="cpu")
            agent = DQNAgent(state_size, action_size, agent_config)
            print("Detected standard DQN agent from checkpoint")
    except Exception as e:
        print(f"Warning: Could not detect agent type, defaulting to standard DQN: {e}")
        agent_config = DQNConfig(device="cpu")
        agent = DQNAgent(state_size, action_size, agent_config)

    # Load trained weights
    print(f"\nLoading agent weights from: {args.load_path}")
    try:
        agent.load(args.load_path)
        print("✓ Agent weights loaded successfully")
    except Exception as e:
        print(f"Error loading agent weights: {e}", file=sys.stderr)
        env.close()
        return 1

    if args.osm_file:
        print(f"Traffic light ID: {env.artifacts.tls_id}")
        print(f"Incoming lanes: {len(env.artifacts.incoming_lanes)}")
        print(f"Outgoing lanes: {len(env.artifacts.outgoing_lanes)}")

    # Run evaluation episodes
    print(f"\n{'=' * 80}")
    if args.no_gui:
        print(f"Running {args.episodes} evaluation episodes (headless mode)...")
    else:
        print(f"Running {args.episodes} evaluation episodes with GUI...")
        print("Watch the SUMO GUI window to see the agent in action!")
        print("Close the SUMO GUI window when done.")
    print("=" * 80)

    total_reward = 0.0
    total_queues = 0.0
    total_waiting = 0.0

    for episode in range(1, args.episodes + 1):
        observation, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_queues = []
        episode_waiting = []

        while not done:
            action = agent.select_action(observation, explore=False)  # No exploration during evaluation
            
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

            episode_reward += reward
            episode_queues.extend(info.get("queues", []))
            episode_waiting.extend(info.get("waiting_times", []))

            observation = next_obs

        total_reward += episode_reward
        total_queues += sum(episode_queues)
        total_waiting += sum(episode_waiting)

        print(
            f"Episode {episode:03d} | Reward: {episode_reward:7.2f} | "
            f"Total Queues: {sum(episode_queues):3d} | "
            f"Total Waiting: {sum(episode_waiting):6.2f}s"
        )

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Episodes: {args.episodes}")
    print(f"Average Reward: {total_reward / args.episodes:.2f}")
    print(f"Average Total Queues: {total_queues / args.episodes:.1f}")
    print(f"Average Total Waiting Time: {total_waiting / args.episodes:.2f}s")
    print("=" * 80)

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))



