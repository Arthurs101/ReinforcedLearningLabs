"""Centralized multi-TLS environment for single agent controlling all traffic lights."""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from .environment import SUMOEnvironment, SUMOEnvironmentConfig
from .scenario import OSMScenario


class CentralizedMultiTLSEnvironment:
    """Single agent controlling multiple traffic lights simultaneously."""

    def __init__(
        self,
        config: Optional[SUMOEnvironmentConfig] = None,
        scenario: Optional[OSMScenario] = None,
        tls_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize centralized multi-TLS environment.

        Args:
            config: Environment configuration
            scenario: OSM scenario (must be OSMScenario)
            tls_ids: List of traffic light IDs to control. If None, controls all TLS in the network.
        """
        self.config = config or SUMOEnvironmentConfig()
        
        if scenario is None:
            raise ValueError("scenario must be provided for centralized multi-TLS environment")
        if not isinstance(scenario, OSMScenario):
            raise ValueError("Centralized multi-TLS environment only supports OSMScenario")
        
        self.scenario = scenario
        self.scenario_builder = scenario
        
        # Build scenario to get network file
        artifacts = self.scenario_builder.build()
        net_file = artifacts.net_file
        
        # Get all TLS lanes
        if tls_ids is None:
            # Get all TLS IDs from the network
            all_tls_lanes = self.scenario.get_all_tls_lanes(net_file)
            self.tls_ids = list(all_tls_lanes.keys())
        else:
            # Use provided TLS IDs
            all_tls_lanes = self.scenario.get_all_tls_lanes(net_file)
            self.tls_ids = [tls_id for tls_id in tls_ids if tls_id in all_tls_lanes]
        
        if not self.tls_ids:
            raise ValueError("No valid traffic lights found for centralized control")
        
        print(f"Centralized control: One agent controlling {len(self.tls_ids)} traffic lights")
        print(f"Traffic light IDs: {self.tls_ids[:10]}{'...' if len(self.tls_ids) > 10 else ''}")
        
        # Store lane mappings for each TLS
        self.tls_lanes = {tls_id: all_tls_lanes[tls_id] for tls_id in self.tls_ids}
        
        # Create a shared environment that will be used for TraCI connection
        self.shared_env = SUMOEnvironment(config=self.config, scenario=self.scenario)
        
        # Initialize phase maps and action spaces for each TLS
        self.tls_phase_maps: Dict[str, Tuple[str, ...]] = {}
        self.tls_action_sizes: Dict[str, int] = {}
        self.tls_current_phases: Dict[str, int] = {}
        self.tls_phase_durations: Dict[str, int] = {}
        self.tls_last_actions: Dict[str, int] = {}
        self.tls_prev_queues: Dict[str, int] = {}
        
        # Calculate total observation and action space sizes
        self._initialize_tls_configs()
        
        # Set up action space: MultiDiscrete for each TLS
        if self.config.enable_duration_control:
            # Each TLS: [num_phases, num_durations]
            action_nvecs = []
            for tls_id in self.tls_ids:
                num_phases = len(self.tls_phase_maps[tls_id])
                num_durations = len(self.config.duration_options)
                action_nvecs.extend([num_phases, num_durations])
            self.action_space = gym.spaces.MultiDiscrete(action_nvecs)
        else:
            # Each TLS: num_phases
            action_nvecs = [len(self.tls_phase_maps[tls_id]) for tls_id in self.tls_ids]
            self.action_space = gym.spaces.MultiDiscrete(action_nvecs)
        
        # Observation space: concatenation of all TLS observations
        total_obs_size = sum(
            len(self.tls_lanes[tls_id][0]) * 3  # 3 features per lane (queue, speed, wait)
            for tls_id in self.tls_ids
        )
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(total_obs_size,),
            dtype=np.float32,
        )
        
        print(f"Action space: MultiDiscrete {self.action_space.nvec}")
        print(f"Observation space: Box({total_obs_size},)")

    def _initialize_tls_configs(self) -> None:
        """Initialize phase maps and configurations for each TLS."""
        import traci
        
        # Start TraCI connection temporarily to get phase maps
        if not self.shared_env._conn_active:
            self.shared_env._start_traci(self.config.seed, force_gui=False)
            temp_connection = True
        else:
            temp_connection = False
        
        try:
            for tls_id in self.tls_ids:
                try:
                    current_state = traci.trafficlight.getRedYellowGreenState(tls_id)
                    
                    if self.config.extract_phases_from_sumo:
                        try:
                            tls_program = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
                            if tls_program and len(tls_program) > 0:
                                valid_phases = []
                                for phase in tls_program[0].phases:
                                    state = phase.state
                                    if 'G' in state and state.count('G') >= 2:
                                        valid_phases.append(state)
                                
                                if valid_phases:
                                    self.tls_phase_maps[tls_id] = tuple(valid_phases)
                                else:
                                    self.tls_phase_maps[tls_id] = self._generate_phase_map(current_state)
                            else:
                                self.tls_phase_maps[tls_id] = self._generate_phase_map(current_state)
                        except Exception:
                            self.tls_phase_maps[tls_id] = self._generate_phase_map(current_state)
                    else:
                        self.tls_phase_maps[tls_id] = self._generate_phase_map(current_state)
                    
                    self.tls_current_phases[tls_id] = 0
                    self.tls_phase_durations[tls_id] = 0
                    self.tls_last_actions[tls_id] = 0
                    self.tls_prev_queues[tls_id] = 0
                    
                except Exception as e:
                    print(f"Warning: Could not initialize TLS {tls_id}: {e}", file=sys.stderr)
                    # Fallback to default
                    self.tls_phase_maps[tls_id] = ("GGrr", "rrGG")
                    self.tls_current_phases[tls_id] = 0
                    self.tls_phase_durations[tls_id] = 0
                    self.tls_last_actions[tls_id] = 0
                    self.tls_prev_queues[tls_id] = 0
        finally:
            if temp_connection:
                traci.close(False)
                self.shared_env._conn_active = False

    def _generate_phase_map(self, current_state: str) -> Tuple[str, ...]:
        """Generate phase map based on number of links."""
        num_links = len(current_state)
        
        if num_links == 4:
            return ("GGrr", "rrGG")
        elif num_links >= 2:
            half = num_links // 2
            phase1 = "G" * half + "r" * (num_links - half)
            phase2 = "r" * half + "G" * (num_links - half)
            return (phase1, phase2)
        else:
            return (current_state,)

    def reset(self, *, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment and return observation for all TLS."""
        obs, _ = self.shared_env.reset(seed=seed)
        
        # Reset TLS states
        for tls_id in self.tls_ids:
            self.tls_current_phases[tls_id] = 0
            self.tls_phase_durations[tls_id] = 0
            self.tls_last_actions[tls_id] = 0
            self.tls_prev_queues[tls_id] = 0
        
        # Get observation for all TLS
        observation = self._collect_observation()
        info = self._collect_info()
        
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Step environment with actions for all TLS.

        Args:
            action: Array of actions, one per TLS (or [phase, duration] pairs if duration control enabled)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        import traci
        
        # Apply actions to all TLS
        action_idx = 0
        for tls_id in self.tls_ids:
            if self.config.enable_duration_control:
                phase_idx = int(action[action_idx])
                duration_idx = int(action[action_idx + 1])
                action_idx += 2
                
                duration_steps = int(self.config.duration_options[duration_idx] / self.config.step_length)
                
                if phase_idx != self.tls_current_phases[tls_id] or self.tls_phase_durations[tls_id] <= 0:
                    if self.tls_phase_durations[tls_id] > 0 and self.tls_phase_durations[tls_id] < int(self.config.min_green_time / self.config.step_length):
                        self.tls_phase_durations[tls_id] = int(self.config.min_green_time / self.config.step_length)
                    else:
                        state = self.tls_phase_maps[tls_id][phase_idx]
                        traci.trafficlight.setRedYellowGreenState(tls_id, state)
                        self.tls_current_phases[tls_id] = phase_idx
                        self.tls_phase_durations[tls_id] = duration_steps
                        self.tls_last_actions[tls_id] = phase_idx * len(self.config.duration_options) + duration_idx
                else:
                    self.tls_phase_durations[tls_id] -= 1
            else:
                phase_idx = int(action[action_idx])
                action_idx += 1
                
                if phase_idx != self.tls_current_phases[tls_id]:
                    state = self.tls_phase_maps[tls_id][phase_idx]
                    traci.trafficlight.setRedYellowGreenState(tls_id, state)
                    self.tls_current_phases[tls_id] = phase_idx
                    self.tls_last_actions[tls_id] = phase_idx
        
        # Step the shared simulation
        obs, reward, terminated, truncated, info = self.shared_env.step(0)  # Dummy action
        
        # Collect observation and calculate reward
        observation = self._collect_observation()
        total_reward = self._calculate_reward()
        info = self._collect_info()
        
        return observation, total_reward, terminated, truncated, info

    def _collect_observation(self) -> np.ndarray:
        """Collect observation from all TLS lanes."""
        import traci
        
        features: List[float] = []
        
        for tls_id in self.tls_ids:
            incoming_lanes = self.tls_lanes[tls_id][0]
            
            queues = [int(traci.lane.getLastStepHaltingNumber(lane)) for lane in incoming_lanes]
            speeds = [float(traci.lane.getLastStepMeanSpeed(lane)) for lane in incoming_lanes]
            waits = [float(traci.lane.getWaitingTime(lane)) for lane in incoming_lanes]
            
            for queue, speed, wait in zip(queues, speeds, waits):
                queue_norm = min(queue / max(self.config.queue_capacity, 1e-6), 1.0)
                speed_norm = min(speed / max(self.config.max_speed, 1e-6), 1.0)
                wait_norm = min(wait / max(self.config.max_wait_time, 1e-6), 1.0)
                features.extend([queue_norm, speed_norm, wait_norm])
        
        return np.asarray(features, dtype=np.float32)

    def _calculate_reward(self) -> float:
        """Calculate total reward across all TLS."""
        import traci
        
        total_reward = 0.0
        
        for tls_id in self.tls_ids:
            incoming_lanes = self.tls_lanes[tls_id][0]
            
            queues = [int(traci.lane.getLastStepHaltingNumber(lane)) for lane in incoming_lanes]
            waits = [float(traci.lane.getWaitingTime(lane)) for lane in incoming_lanes]
            
            current_total_queues = sum(queues)
            queue_penalty = self.config.queue_penalty * current_total_queues
            wait_penalty = self.config.wait_penalty * sum(waits)
            reward = -(queue_penalty + wait_penalty)
            
            if queue_penalty == 0:
                reward += self.config.clear_bonus
            
            queue_reduction = self.tls_prev_queues[tls_id] - current_total_queues
            if queue_reduction > 0:
                reward += 0.1 * queue_reduction
            
            self.tls_prev_queues[tls_id] = current_total_queues
            total_reward += reward
        
        return float(total_reward)

    def _collect_info(self) -> Dict:
        """Collect info for all TLS."""
        import traci
        
        info = {
            "num_tls": len(self.tls_ids),
            "tls_info": {},
        }
        
        for tls_id in self.tls_ids:
            incoming_lanes = self.tls_lanes[tls_id][0]
            queues = [int(traci.lane.getLastStepHaltingNumber(lane)) for lane in incoming_lanes]
            waits = [float(traci.lane.getWaitingTime(lane)) for lane in incoming_lanes]
            
            info["tls_info"][tls_id] = {
                "action": self.tls_last_actions[tls_id],
                "queues": queues,
                "waiting_times": waits,
                "current_phase": self.tls_current_phases[tls_id],
            }
            
            if self.config.enable_duration_control:
                info["tls_info"][tls_id]["phase_duration_remaining"] = self.tls_phase_durations[tls_id]
        
        return info

    def close(self) -> None:
        """Close the shared environment."""
        self.shared_env.close()

