"""Command-line utility to train or evaluate agents on the SUMO environment."""

from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Union

import numpy as np
import torch

from agents import (
    AgentTransition,
    DQNAgent,
    DQNConfig,
    ImprovedDQNAgent,
    ImprovedDQNConfig,
)
from sumo import SUMOEnvironment, SUMOEnvironmentConfig, OSMScenario, SimpleIntersectionScenario, MultiAgentSUMOEnvironment, CentralizedMultiTLSEnvironment
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
    parser.add_argument("--flow-rate", type=int, default=500, help="Vehicle flow rate per hour (for OSM scenarios, default: 500)")
    parser.add_argument("--tls-id", type=str, default=None, help="Traffic light ID to control (for OSM scenarios, defaults to first TLS found)")
    parser.add_argument("--agent-type", type=str, default="dqn", choices=["dqn", "improved"], help="Agent type: 'dqn' (standard) or 'improved' (Double DQN + Dueling)")
    parser.add_argument("--multi-agent", action="store_true", help="Enable multi-agent mode: one agent per traffic light (for OSM scenarios)")
    parser.add_argument("--centralized", action="store_true", help="Enable centralized mode: one agent controlling all traffic lights (for OSM scenarios)")
    parser.add_argument("--max-tls", type=int, default=None, help="Maximum number of traffic lights to control (default: all)")
    parser.add_argument("--tls-per-agent", type=int, default=1, help="Number of traffic lights per agent in multi-agent mode (default: 1, use --group-by-proximity to auto-group)")
    parser.add_argument("--group-by-proximity", action="store_true", help="Automatically group nearby traffic lights (for multi-agent mode)")
    parser.add_argument("--max-group-distance", type=float, default=None, help="Maximum distance (meters) for grouping TLS by proximity")
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


def run_multi_agent_episode(
    env: MultiAgentSUMOEnvironment,
    agents: Dict[str, Union[DQNAgent, ImprovedDQNAgent]],
    train: bool = True
) -> Tuple[Dict[str, float], Dict[str, dict]]:
    """Run one episode with multiple agents."""
    observations = env.reset()
    done = False
    episode_rewards = {tls_id: 0.0 for tls_id in agents.keys()}
    episode_infos = {tls_id: None for tls_id in agents.keys()}
    episode_losses = {tls_id: [] for tls_id in agents.keys()}
    episode_actions = {tls_id: [] for tls_id in agents.keys()}

    while not done:
        # Each agent selects an action based on its observation
        actions = {}
        for agent_id, agent in agents.items():
            obs = observations[agent_id]
            action = agent.select_action(obs, explore=train)
            
            # For MultiDiscrete action spaces (groups), keep action as-is
            # The environment wrapper will decode it
            action_space = env.get_action_spaces()[agent_id]
            if isinstance(action_space, gym.spaces.MultiDiscrete):
                # For grouped agents, action is already flattened integer
                # Pass it directly - the wrapper will decode it
                actions[agent_id] = action
            else:
                # Single TLS, discrete action space
                actions[agent_id] = action
            
            episode_actions[agent_id].append(action)

        # Step environment with all actions
        next_obs, rewards, dones_dict, infos = env.step(actions)
        
        # Check if any agent's episode is done
        done = any(dones_dict.values())
        
        # Update each agent
        for tls_id, agent in agents.items():
            if train:
                transition = AgentTransition(
                    state=observations[tls_id],
                    action=actions[tls_id],
                    reward=rewards[tls_id],
                    next_state=next_obs[tls_id],
                    done=dones_dict[tls_id],
                )
                agent.observe(transition)
                update_metrics = agent.update()
                if update_metrics and "loss" in update_metrics:
                    episode_losses[tls_id].append(update_metrics["loss"])
            
            episode_rewards[tls_id] += rewards[tls_id]
            episode_infos[tls_id] = infos[tls_id]
        
        observations = next_obs

    # Calculate action distributions
    episode_info_dict = {}
    for tls_id in agents.keys():
        action_counts = {}
        for action in episode_actions[tls_id]:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        episode_info_dict[tls_id] = {
            "losses": episode_losses[tls_id],
            "actions": episode_actions[tls_id],
            "action_distribution": action_counts,
            "avg_loss": float(np.mean(episode_losses[tls_id])) if episode_losses[tls_id] else None,
            "num_updates": len(episode_losses[tls_id]),
            **episode_infos[tls_id],
        }

    return episode_rewards, episode_info_dict


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

    # Check if centralized mode (one agent controlling all TLS)
    if args.centralized:
        if not args.osm_file:
            print("Error: --centralized requires --osm-file", file=sys.stderr)
            return 1
        
        # Disable duration control for centralized mode (action space too large otherwise)
        centralized_config = SUMOEnvironmentConfig(
            max_steps=args.max_steps,
            step_length=args.step_length,
            use_gui=args.use_gui,
            seed=args.seed,
            enable_duration_control=False,  # Disable to keep action space manageable
            extract_phases_from_sumo=env_config.extract_phases_from_sumo,
        )
        
        # Create centralized environment
        end_time = int(args.max_steps * args.step_length)
        scenario = OSMScenario(
            osm_file=args.osm_file,
            flow_rate=args.flow_rate,
            tls_id=None,  # Will use all TLS
            begin_time=0,
            end_time=end_time,
        )
        
        # Limit number of TLS if specified
        tls_ids = None
        if args.max_tls:
            # Get all TLS IDs first
            artifacts = scenario.build()
            all_tls_lanes = scenario.get_all_tls_lanes(artifacts.net_file)
            tls_ids = list(all_tls_lanes.keys())[:args.max_tls]
            print(f"Limiting to {len(tls_ids)} traffic lights (--max-tls={args.max_tls})")
        
        centralized_env = CentralizedMultiTLSEnvironment(
            config=centralized_config,
            scenario=scenario,
            tls_ids=tls_ids,
        )
        
        # Create single agent for all TLS
        obs_size = centralized_env.observation_space.shape[0]
        
        # For centralized mode without duration control, action space is MultiDiscrete with one action per TLS
        # We'll use the sum of max actions per TLS as a reasonable approximation
        # Actually, we should use the product, but that's still too large. Let's use a different approach.
        if isinstance(centralized_env.action_space, gym.spaces.MultiDiscrete):
            # Use sum instead of product to keep it manageable
            # Each TLS gets its own action branch in the network
            # For now, use max action size per TLS as a proxy
            max_actions_per_tls = max(centralized_env.action_space.nvec)
            # Use a reasonable upper bound: sum of all action sizes
            action_size = sum(centralized_env.action_space.nvec)
            print(f"Warning: Centralized mode with {len(centralized_env.tls_ids)} TLS creates large action space.")
            print(f"  Using sum-based action size: {action_size} (instead of product: {np.prod(centralized_env.action_space.nvec)})")
            print(f"  Consider using --max-tls to limit the number of TLS controlled.")
        else:
            action_size = centralized_env.action_space.n
        
        if args.agent_type == "improved":
            agent = ImprovedDQNAgent(obs_size, action_size, agent_config)
        else:
            agent = DQNAgent(obs_size, action_size, agent_config)
        
        print(f"\nCentralized control: One agent controlling {len(centralized_env.tls_ids)} traffic lights")
        print(f"Observation size: {obs_size}")
        print(f"Action size: {action_size} (MultiDiscrete {centralized_env.action_space.nvec})")
        
        # Centralized training
        print("\nStarting centralized training...")
        print("=" * 80)
        
        training_metrics = {
            "run_id": run_id,
            "mode": "centralized_training",
            "episodes": args.episodes,
            "num_tls": len(centralized_env.tls_ids),
            "tls_ids": centralized_env.tls_ids,
            "episode_data": [],
            "timestamp": datetime.now().isoformat(),
        }
        
        for episode in range(1, args.episodes + 1):
            observation, _ = centralized_env.reset()
            done = False
            episode_reward = 0.0
            episode_info = None
            episode_losses = []
            episode_actions = []
            
            while not done:
                # Update epsilon for ImprovedDQNAgent (it's normally updated in select_action)
                if isinstance(agent, ImprovedDQNAgent):
                    # Manually update epsilon based on steps (same logic as in select_action)
                    epsilon = max(
                        agent.config.epsilon_end,
                        agent.config.epsilon_start - (agent.config.epsilon_start - agent.config.epsilon_end) 
                        * (agent._steps / agent.config.epsilon_decay_steps)
                    )
                    agent.epsilon = epsilon
                
                # For MultiDiscrete, we need to select actions for each TLS independently
                if isinstance(centralized_env.action_space, gym.spaces.MultiDiscrete):
                    # Select action for each TLS dimension independently
                    action_array = np.zeros(len(centralized_env.action_space.nvec), dtype=np.int64)
                    obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=agent.device).unsqueeze(0)
                    
                    with torch.no_grad():
                        q_values = agent.policy_net(obs_tensor)
                    
                    # For each TLS, select action from its action space
                    # Split Q-values proportionally to each TLS's action space size
                    q_start_idx = 0
                    for i, n_actions in enumerate(centralized_env.action_space.nvec):
                        # Calculate how many Q-values this TLS should get
                        # Use proportional allocation based on action space sizes
                        total_actions = sum(centralized_env.action_space.nvec)
                        q_allocation = int((n_actions / total_actions) * q_values.shape[1])
                        q_allocation = max(1, min(q_allocation, n_actions))  # At least 1, at most n_actions
                        
                        end_idx = min(q_start_idx + q_allocation, q_values.shape[1])
                        
                        if agent.epsilon > 0 and random.random() < agent.epsilon:
                            action_array[i] = random.randrange(n_actions)
                        else:
                            if end_idx > q_start_idx:
                                # Use the Q-values allocated to this TLS
                                tls_q_values = q_values[0, q_start_idx:end_idx]
                                # If we have fewer Q-values than actions, pad or use modulo
                                if q_allocation < n_actions:
                                    # Use argmax on available Q-values, then map to action space
                                    selected_q_idx = int(torch.argmax(tls_q_values).item())
                                    # Map proportionally to action space
                                    action_array[i] = int((selected_q_idx / q_allocation) * n_actions)
                                    action_array[i] = min(action_array[i], n_actions - 1)
                                else:
                                    action_array[i] = int(torch.argmax(tls_q_values).item())
                            else:
                                action_array[i] = random.randrange(n_actions)
                        
                        q_start_idx = end_idx
                    
                    env_action = action_array
                    # Store as a tuple for metrics
                    action_for_storage = tuple(action_array)
                else:
                    action = agent.select_action(observation, explore=True)
                    env_action = action
                    action_for_storage = action
                
                next_obs, reward, terminated, truncated, info = centralized_env.step(env_action)
                done = terminated or truncated
                episode_info = info
                episode_actions.append(action_for_storage)
                
                # For training, we need to convert MultiDiscrete action to a single integer
                # Use a hash-based encoding that maps to the action_size range
                if isinstance(centralized_env.action_space, gym.spaces.MultiDiscrete):
                    # Use hash-based encoding to map action tuple to action_size range
                    # This ensures reasonable distribution across the action space
                    # Use a stable hash (tuple hash) and map to action_size
                    action_hash = hash(action_for_storage)
                    # Map hash to action_size range (use absolute value and modulo)
                    encoded_action = abs(action_hash) % agent.action_size
                else:
                    encoded_action = action_for_storage
                
                transition = AgentTransition(
                    state=observation,
                    action=encoded_action,
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
            
            total_queues = sum(sum(tls_info.get("queues", [])) for tls_info in episode_info.get("tls_info", {}).values())
            total_waiting = sum(sum(tls_info.get("waiting_times", [])) for tls_info in episode_info.get("tls_info", {}).values())
            
            episode_data = {
                "episode": episode,
                "reward": float(episode_reward),
                "epsilon": float(agent.epsilon),
                "total_queues": int(total_queues),
                "total_waiting_time": float(total_waiting),
                "avg_loss": float(np.mean(episode_losses)) if episode_losses else None,
                "num_updates": len(episode_losses),
                "action_distribution": action_counts,
                "tls_info": episode_info.get("tls_info", {}),
            }
            training_metrics["episode_data"].append(episode_data)
            
            print(f"Episode {episode:03d} | Reward: {episode_reward:7.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Total Queues: {total_queues:3d} | "
                  f"Total Waiting: {total_waiting:6.2f}s")
        
        # Save agent
        weights_path = args.save_path or (run_dir / "weights.pt")
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(weights_path)
        print(f"\nSaved centralized agent to {weights_path}")
        
        # Save metrics
        save_metrics(run_dir, training_metrics)
        
        print(f"\nCentralized training complete! Run ID: {run_id}")
        print(f"  Weights: {weights_path}")
        print(f"  Metrics: {run_dir / 'metrics.json'}")
        
        centralized_env.close()
        return 0

    # Check if multi-agent mode
    if args.multi_agent:
        if not args.osm_file:
            print("Error: --multi-agent requires --osm-file", file=sys.stderr)
            return 1
        
        # Create multi-agent environment
        end_time = int(args.max_steps * args.step_length)
        scenario = OSMScenario(
            osm_file=args.osm_file,
            flow_rate=args.flow_rate,
            tls_id=None,  # Will use all TLS
            begin_time=0,
            end_time=end_time,
        )
        
        # Limit number of TLS if specified
        tls_ids = None
        if args.max_tls:
            # Get all TLS IDs first
            artifacts = scenario.build()
            all_tls_lanes = scenario.get_all_tls_lanes(artifacts.net_file)
            tls_ids = list(all_tls_lanes.keys())[:args.max_tls]
            print(f"Limiting to {len(tls_ids)} traffic lights (--max-tls={args.max_tls})")
        
        # Group TLS if requested
        tls_groups = None
        if args.group_by_proximity or args.tls_per_agent > 1:
            artifacts = scenario.build()
            net_file = artifacts.net_file
            
            if args.group_by_proximity:
                # Group by proximity
                tls_groups = scenario.group_tls_by_proximity(
                    net_file=net_file,
                    tls_ids=tls_ids,
                    max_tls_per_group=args.tls_per_agent,
                    max_distance=args.max_group_distance,
                )
                print(f"Grouped {len([tls for group in tls_groups for tls in group])} TLS into {len(tls_groups)} groups by proximity")
            else:
                # Simple sequential grouping
                if tls_ids is None:
                    all_tls_lanes = scenario.get_all_tls_lanes(net_file)
                    tls_ids = list(all_tls_lanes.keys())
                
                tls_groups = []
                for i in range(0, len(tls_ids), args.tls_per_agent):
                    tls_groups.append(tls_ids[i:i + args.tls_per_agent])
                print(f"Grouped {len(tls_ids)} TLS into {len(tls_groups)} groups ({args.tls_per_agent} TLS per agent)")
        
        multi_env = MultiAgentSUMOEnvironment(
            config=env_config,
            scenario=scenario,
            tls_ids=tls_ids,
            tls_groups=tls_groups,
        )
        
        # Create agents (one per group or per TLS)
        agents: Dict[str, Union[DQNAgent, ImprovedDQNAgent]] = {}
        action_spaces = multi_env.get_action_spaces()
        observation_spaces = multi_env.get_observation_spaces()
        
        agent_ids = list(multi_env.envs.keys())  # These are either TLS IDs or group IDs
        
        print(f"\nCreating {len(agent_ids)} agents...")
        for agent_id in agent_ids:
            obs_size = observation_spaces[agent_id].shape[0]
            
            if isinstance(action_spaces[agent_id], gym.spaces.MultiDiscrete):
                # Flatten MultiDiscrete action space
                action_size = np.prod(action_spaces[agent_id].nvec)
            else:
                action_size = action_spaces[agent_id].n
            
            if args.agent_type == "improved":
                agents[agent_id] = ImprovedDQNAgent(obs_size, action_size, agent_config)
            else:
                agents[agent_id] = DQNAgent(obs_size, action_size, agent_config)
            
            if agent_id.startswith("group_"):
                group_idx = int(agent_id.split("_")[1])
                tls_in_group = multi_env.tls_groups[group_idx]
                print(f"  Agent {agent_id}: {len(tls_in_group)} TLS - obs_size={obs_size}, action_size={action_size}")
            else:
                print(f"  Agent for TLS {agent_id}: obs_size={obs_size}, action_size={action_size}")
        
        # Multi-agent training
        print("\nStarting multi-agent training...")
        print("=" * 80)
        
        training_metrics = {
            "run_id": run_id,
            "mode": "multi_agent_training",
            "episodes": args.episodes,
            "num_agents": len(agents),
            "tls_ids": list(agents.keys()),
            "episode_data": [],
            "timestamp": datetime.now().isoformat(),
        }
        
        for episode in range(1, args.episodes + 1):
            rewards, infos = run_multi_agent_episode(multi_env, agents, train=True)
            
            total_reward = sum(rewards.values())
            total_queues = 0
            total_waiting = 0
            
            # Collect queues and waiting times (handle both individual TLS and groups)
            for info in infos.values():
                if "tls_info" in info:
                    # Grouped agent - sum across all TLS in group
                    for tls_info in info["tls_info"].values():
                        total_queues += sum(tls_info.get("queues", []))
                        total_waiting += sum(tls_info.get("waiting_times", []))
                else:
                    # Individual TLS agent
                    total_queues += sum(info.get("queues", []))
                    total_waiting += sum(info.get("waiting_times", []))
            
            # Calculate average epsilon across all agents
            avg_epsilon = float(np.mean([agent.epsilon for agent in agents.values()]))
            min_epsilon = float(np.min([agent.epsilon for agent in agents.values()]))
            max_epsilon = float(np.max([agent.epsilon for agent in agents.values()]))
            
            episode_data = {
                "episode": episode,
                "total_reward": float(total_reward),
                "avg_reward_per_agent": float(total_reward / len(agents)),
                "total_queues": int(total_queues),
                "total_waiting_time": float(total_waiting),
                "avg_epsilon": avg_epsilon,
                "min_epsilon": min_epsilon,
                "max_epsilon": max_epsilon,
                "agent_rewards": {tls_id: float(r) for tls_id, r in rewards.items()},
                "agent_epsilon": {agent_id: float(agent.epsilon) for agent_id, agent in agents.items()},
                "agent_metrics": {
                    agent_id: {
                        "queues": info.get("queues", []) if "tls_info" not in info else sum((tls_info.get("queues", []) for tls_info in info.get("tls_info", {}).values()), []),
                        "waiting_times": info.get("waiting_times", []) if "tls_info" not in info else sum((tls_info.get("waiting_times", []) for tls_info in info.get("tls_info", {}).values()), []),
                        "avg_loss": info.get("avg_loss"),
                        "num_updates": info.get("num_updates", 0),
                        "tls_info": info.get("tls_info", {}),  # Include TLS info for groups
                    }
                    for agent_id, info in infos.items()
                },
            }
            training_metrics["episode_data"].append(episode_data)
            
            print(f"Episode {episode:03d} | Total Reward: {total_reward:7.2f} | "
                  f"Avg/Agent: {total_reward/len(agents):7.2f} | "
                  f"Epsilon: {avg_epsilon:.3f} | "
                  f"Total Queues: {total_queues:3d} | "
                  f"Total Waiting: {total_waiting:6.2f}s")
        
        # Save agents
        for tls_id, agent in agents.items():
            weights_path = run_dir / f"weights_{tls_id}.pt"
            agent.save(weights_path)
            print(f"Saved agent for TLS {tls_id} to {weights_path}")
        
        # Save metrics
        save_metrics(run_dir, training_metrics)
        
        print(f"\nMulti-agent training complete! Run ID: {run_id}")
        print(f"  Agents: {len(agents)}")
        print(f"  Metrics: {run_dir / 'metrics.json'}")
        
        multi_env.close()
        return 0
    
    # Single-agent mode (original code)
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

