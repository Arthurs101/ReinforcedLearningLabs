"""Visualize a trained agent controlling traffic lights in SUMO GUI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from agents import DQNAgent, DQNConfig, ImprovedDQNAgent, ImprovedDQNConfig
from sumo import (
    SUMOEnvironment,
    SUMOEnvironmentConfig,
    OSMScenario,
    SimpleIntersectionScenario,
    MultiAgentSUMOEnvironment,
)
import gymnasium as gym
import numpy as np


DEFAULT_FLOW_RATE = 600
DEFAULT_MAX_STEPS = 900
DEFAULT_STEP_LENGTH = 1.0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a trained DQN agent controlling traffic lights in SUMO GUI"
    )
    parser.add_argument(
        "--load-path",
        type=Path,
        default=None,
        help="Path to trained agent weights file (e.g., output/run_id/weights.pt). Optional if --run-dir is provided.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to a training run directory (loads config, metrics, and weights automatically).",
    )
    parser.add_argument(
        "--multi-agent",
        action="store_true",
        help="Visualize a multi-agent run by loading all agent checkpoints in the run directory.",
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


def load_run_artifacts(run_dir: Path) -> Tuple[Optional[dict], Optional[dict], Dict[str, Path], Optional[Path]]:
    """Load config, metrics, and weights from a run directory."""
    run_config = None
    metrics = None
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"

    if config_path.exists():
        with open(config_path) as f:
            run_config = json.load(f)
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    multi_agent_weights: Dict[str, Path] = {}
    for weight_path in sorted(run_dir.glob("weights_*.pt")):
        agent_id = weight_path.stem.replace("weights_", "")
        multi_agent_weights[agent_id] = weight_path

    single_agent_weight = run_dir / "weights.pt"
    if not single_agent_weight.exists():
        single_agent_weight = None

    return run_config, metrics, multi_agent_weights, single_agent_weight


def apply_run_config_defaults(args: argparse.Namespace, run_config: Optional[dict]) -> None:
    """Populate CLI arguments with values stored in the run config."""
    if not run_config:
        return

    osm_file = run_config.get("osm_file")
    if args.osm_file is None and osm_file:
        args.osm_file = Path(osm_file)
        print(f"Using OSM file from run config: {args.osm_file}")

    if args.flow_rate == DEFAULT_FLOW_RATE and run_config.get("flow_rate") is not None:
        args.flow_rate = int(run_config["flow_rate"])
        print(f"Using flow rate from run config: {args.flow_rate}")

    if args.max_steps == DEFAULT_MAX_STEPS and run_config.get("max_steps") is not None:
        args.max_steps = int(run_config["max_steps"])

    if args.step_length == DEFAULT_STEP_LENGTH and run_config.get("step_length") is not None:
        args.step_length = float(run_config["step_length"])

    if args.seed is None and run_config.get("seed") is not None:
        args.seed = run_config["seed"]


def detect_agent_type(weight_path: Path) -> str:
    """Return 'improved' or 'dqn' based on checkpoint metadata."""
    try:
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config")
        if config and hasattr(config, "use_double_dqn"):
            return "improved"
    except Exception as exc:
        print(f"Warning: Could not detect agent type from {weight_path}: {exc}")
    return "dqn"


def build_environment_config(args: argparse.Namespace) -> SUMOEnvironmentConfig:
    return SUMOEnvironmentConfig(
        max_steps=args.max_steps,
        step_length=args.step_length,
        use_gui=not args.no_gui,
        seed=args.seed,
    )


def build_scenario(args: argparse.Namespace) -> Tuple[OSMScenario | SimpleIntersectionScenario, bool]:
    if args.osm_file:
        if not args.osm_file.exists():
            raise FileNotFoundError(f"OSM file not found: {args.osm_file}")
        end_time = int(args.max_steps * args.step_length)
        scenario = OSMScenario(
            osm_file=args.osm_file,
            flow_rate=args.flow_rate,
            tls_id=args.tls_id,
            begin_time=0,
            end_time=end_time,
        )
        return scenario, True

    scenario = SimpleIntersectionScenario()
    return scenario, False


def infer_tls_groups(metrics: Optional[dict]) -> List[List[str]]:
    """Infer TLS grouping from multi-agent training metrics."""
    if not metrics or metrics.get("mode") != "multi_agent_training":
        return []

    for episode in metrics.get("episode_data", []):
        agent_metrics = episode.get("agent_metrics")
        if not agent_metrics:
            continue

        groups: List[List[str]] = []
        for agent_id in sorted(agent_metrics.keys()):
            info = agent_metrics[agent_id]
            if isinstance(info, dict) and "tls_info" in info and info["tls_info"]:
                groups.append(list(info["tls_info"].keys()))
            elif isinstance(info, dict) and info.get("tls_id"):
                groups.append([info["tls_id"]])
            else:
                groups.append([])

        if any(group for group in groups):
            return groups

    return []


def create_agent(agent_type: str, obs_size: int, action_size: int) -> tuple[DQNAgent | ImprovedDQNAgent, str]:
    if agent_type == "improved":
        agent_config = ImprovedDQNConfig(device="cpu")
        return ImprovedDQNAgent(obs_size, action_size, agent_config), agent_type

    agent_config = DQNConfig(device="cpu")
    return DQNAgent(obs_size, action_size, agent_config), "dqn"


def summarize_multi_agent_episode(infos: Dict[str, dict]) -> Tuple[int, float]:
    """Return total queues and waiting time from final step infos."""
    total_queues = 0
    total_waiting = 0.0

    for info in infos.values():
        if "tls_info" in info:
            for tls_info in info["tls_info"].values():
                total_queues += sum(tls_info.get("queues", []))
                total_waiting += float(sum(tls_info.get("waiting_times", [])))
    else:
            total_queues += sum(info.get("queues", []))
            total_waiting += float(sum(info.get("waiting_times", [])))

    return total_queues, total_waiting


def run_single_agent_visualization(args: argparse.Namespace) -> int:
    if not args.load_path or not args.load_path.exists():
        print(f"Error: Weights file not found: {args.load_path}", file=sys.stderr)
        return 1

    checkpoint_config = load_config_from_checkpoint(args.load_path)
    apply_run_config_defaults(args, checkpoint_config)

    env_config = build_environment_config(args)
    scenario, is_osm = build_scenario(args)

    env = SUMOEnvironment(config=env_config, scenario=scenario)
    state_size = env.observation_space.shape[0]
    
    if isinstance(env.action_space, gym.spaces.MultiDiscrete):
        action_size = int(np.prod(env.action_space.nvec))
    else:
        action_size = env.action_space.n

    agent_type = detect_agent_type(args.load_path)
    agent, agent_type = create_agent(agent_type, state_size, action_size)

    print(f"\nLoading agent weights from: {args.load_path}")
    try:
        agent.load(args.load_path)
        print("✓ Agent weights loaded successfully")
    except Exception as exc:
        print(f"Error loading agent weights: {exc}", file=sys.stderr)
        env.close()
        return 1

    if is_osm:
        print(f"Traffic light ID: {env.artifacts.tls_id}")
        print(f"Incoming lanes: {len(env.artifacts.incoming_lanes)}")
        print(f"Outgoing lanes: {len(env.artifacts.outgoing_lanes)}")

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
            action = agent.select_action(observation, explore=False)
            
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


def run_multi_agent_visualization(
    args: argparse.Namespace,
    run_config: Optional[dict],
    metrics: Optional[dict],
    weights: Dict[str, Path],
) -> int:
    if not weights:
        print("Error: No multi-agent weight files found. Did you pass the correct run directory?", file=sys.stderr)
        return 1

    apply_run_config_defaults(args, run_config)

    env_config = build_environment_config(args)
    scenario, is_osm = build_scenario(args)

    tls_groups = infer_tls_groups(metrics)
    tls_ids = sorted({tls for group in tls_groups for tls in group}) if tls_groups else None

    multi_env = MultiAgentSUMOEnvironment(
        config=env_config,
        scenario=scenario,
        tls_ids=tls_ids,
        tls_groups=tls_groups if tls_groups else None,
    )

    action_spaces = multi_env.get_action_spaces()
    observation_spaces = multi_env.get_observation_spaces()
    env_agent_ids = list(action_spaces.keys())

    missing_agents = [agent_id for agent_id in env_agent_ids if agent_id not in weights]
    if missing_agents:
        print(f"Error: Missing weights for agents: {missing_agents}", file=sys.stderr)
        multi_env.close()
        return 1

    sample_weight = next(iter(weights.values()))
    agent_type = detect_agent_type(sample_weight)

    agents: Dict[str, DQNAgent | ImprovedDQNAgent] = {}
    for agent_id in env_agent_ids:
        obs_space = observation_spaces[agent_id]
        action_space = action_spaces[agent_id]

        if isinstance(action_space, gym.spaces.MultiDiscrete):
            action_size = int(np.prod(action_space.nvec))
        else:
            action_size = action_space.n

        agent, _ = create_agent(agent_type, obs_space.shape[0], action_size)
        try:
            agent.load(weights[agent_id])
            agent.epsilon = 0.0
        except Exception as exc:
            print(f"Error loading weights for {agent_id}: {exc}", file=sys.stderr)
            multi_env.close()
            return 1

        agents[agent_id] = agent

    print(f"\nLoaded {len(agents)} agents: {', '.join(env_agent_ids)}")
    if is_osm:
        print(f"Scenario: {args.osm_file}")
        print(f"Flow rate: {args.flow_rate} veh/hour")
        print(f"Step length: {args.step_length}s")

    print(f"\n{'=' * 80}")
    if args.no_gui:
        print(f"Running {args.episodes} evaluation episodes (headless mode)...")
    else:
        print(f"Running {args.episodes} evaluation episodes with GUI...")
        print("Watch the SUMO GUI window to see all agents in action!")
        print("Close the SUMO GUI window when done.")
    print("=" * 80)

    total_rewards: List[float] = []
    total_queues: List[int] = []
    total_waiting: List[float] = []

    for episode in range(1, args.episodes + 1):
        observations = multi_env.reset()
        done = False
        episode_agent_rewards = {agent_id: 0.0 for agent_id in agents}
        final_infos: Dict[str, dict] = {}

        while not done:
            actions = {}
            for agent_id, agent in agents.items():
                actions[agent_id] = agent.select_action(observations[agent_id], explore=False)

            next_obs, rewards, dones, infos = multi_env.step(actions)
            final_infos = infos

            for agent_id, reward in rewards.items():
                episode_agent_rewards[agent_id] += reward

            done = any(dones.values())
            observations = next_obs

        episode_total_reward = sum(episode_agent_rewards.values())
        queues, waiting = summarize_multi_agent_episode(final_infos)

        total_rewards.append(episode_total_reward)
        total_queues.append(queues)
        total_waiting.append(waiting)

        avg_per_agent = episode_total_reward / len(agents)
        print(
            f"Episode {episode:03d} | Total Reward: {episode_total_reward:9.2f} | "
            f"Avg/Agent: {avg_per_agent:7.2f} | "
            f"Total Queues: {queues:4d} | "
            f"Total Waiting: {waiting:8.2f}s"
        )

        for agent_id in env_agent_ids:
            print(f"  - {agent_id}: reward={episode_agent_rewards[agent_id]:8.2f}")

    print("\n" + "=" * 80)
    print("MULTI-AGENT EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Episodes: {args.episodes}")
    print(f"Average Total Reward: {np.mean(total_rewards):.2f}")
    print(f"Average Reward per Agent: {np.mean(total_rewards) / len(agents):.2f}")
    print(f"Average Total Queues: {np.mean(total_queues):.1f}")
    print(f"Average Total Waiting Time: {np.mean(total_waiting):.2f}s")
    print("=" * 80)

    multi_env.close()
    return 0


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if not args.load_path and not args.run_dir:
        print("Error: you must provide --load-path or --run-dir", file=sys.stderr)
        return 1

    run_config = None
    run_metrics = None
    multi_agent_weights: Dict[str, Path] = {}
    single_agent_weight: Optional[Path] = None

    if args.run_dir:
        if not args.run_dir.exists():
            print(f"Error: run directory not found: {args.run_dir}", file=sys.stderr)
            return 1
        run_config, run_metrics, multi_agent_weights, single_agent_weight = load_run_artifacts(args.run_dir)
        if args.load_path is None and not args.multi_agent and single_agent_weight:
            args.load_path = single_agent_weight

        if multi_agent_weights and not args.multi_agent:
            args.multi_agent = True

    if args.multi_agent:
        if not args.run_dir:
            if args.load_path:
                args.run_dir = args.load_path.parent
                run_config, run_metrics, multi_agent_weights, _ = load_run_artifacts(args.run_dir)
            else:
                print("Error: multi-agent visualization requires --run-dir", file=sys.stderr)
                return 1

        return run_multi_agent_visualization(args, run_config, run_metrics, multi_agent_weights)

    if args.load_path is None:
        print("Error: single-agent visualization requires --load-path", file=sys.stderr)
        return 1

    return run_single_agent_visualization(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))



