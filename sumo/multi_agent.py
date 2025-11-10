"""Multi-agent environment wrapper for controlling multiple traffic lights."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from .environment import SUMOEnvironment, SUMOEnvironmentConfig
from .scenario import OSMScenario


class MultiAgentSUMOEnvironment:
    """Wrapper for multiple single-agent SUMO environments (one per traffic light or group)."""

    def __init__(
        self,
        config: Optional[SUMOEnvironmentConfig] = None,
        scenario: Optional[OSMScenario] = None,
        tls_ids: Optional[List[str]] = None,
        tls_groups: Optional[List[List[str]]] = None,
    ) -> None:
        """
        Initialize multi-agent environment.

        Args:
            config: Environment configuration
            scenario: OSM scenario (must be OSMScenario, not SimpleIntersectionScenario)
            tls_ids: List of traffic light IDs to control. If None, controls all TLS in the network.
            tls_groups: List of TLS groups, where each group is a list of TLS IDs controlled by one agent.
                       If provided, overrides tls_ids and creates grouped agents.
        """
        self.config = config or SUMOEnvironmentConfig()
        
        if scenario is None:
            raise ValueError("scenario must be provided for multi-agent environment")
        if not isinstance(scenario, OSMScenario):
            raise ValueError("Multi-agent environment only supports OSMScenario")
        
        self.scenario = scenario
        self.scenario_builder = scenario
        
        # Build scenario to get network file
        artifacts = self.scenario_builder.build()
        net_file = artifacts.net_file
        
        # Get all TLS lanes
        all_tls_lanes = self.scenario.get_all_tls_lanes(net_file)
        
        if tls_groups is not None:
            # Use provided groups
            self.tls_groups = [group for group in tls_groups if all(tls_id in all_tls_lanes for tls_id in group)]
            self.tls_ids = [tls_id for group in self.tls_groups for tls_id in group]
            self.use_groups = True
        else:
            # Use individual TLS (one per agent)
            if tls_ids is None:
                self.tls_ids = list(all_tls_lanes.keys())
            else:
                self.tls_ids = [tls_id for tls_id in tls_ids if tls_id in all_tls_lanes]
            
            # Create groups of one TLS each
            self.tls_groups = [[tls_id] for tls_id in self.tls_ids]
            self.use_groups = False
        
        if not self.tls_ids:
            raise ValueError("No valid traffic lights found for multi-agent control")
        
        if self.use_groups:
            print(f"Grouped multi-agent environment: {len(self.tls_groups)} agents controlling {len(self.tls_ids)} traffic lights")
            for i, group in enumerate(self.tls_groups[:5]):
                print(f"  Agent {i}: {len(group)} TLS - {group}")
            if len(self.tls_groups) > 5:
                print(f"  ... and {len(self.tls_groups) - 5} more agents")
        else:
            print(f"Multi-agent environment: Controlling {len(self.tls_ids)} traffic lights")
            print(f"Traffic light IDs: {self.tls_ids[:10]}{'...' if len(self.tls_ids) > 10 else ''}")
        
        # Store lane mappings for each TLS
        self.tls_lanes = {tls_id: all_tls_lanes[tls_id] for tls_id in self.tls_ids}
        
        # Create a shared environment that will be used for TraCI connection
        self.shared_env = SUMOEnvironment(config=self.config, scenario=self.scenario)
        
        # Create environment wrappers
        self.envs: Dict[str, Union[_SingleTLSEnvironmentWrapper, _GroupedTLSEnvironmentWrapper]] = {}
        
        if self.use_groups:
            # Create grouped wrappers
            for group_idx, group in enumerate(self.tls_groups):
                group_id = f"group_{group_idx}"
                group_wrapper = _GroupedTLSEnvironmentWrapper(
                    shared_env=self.shared_env,
                    tls_ids=group,
                    tls_lanes={tls_id: self.tls_lanes[tls_id] for tls_id in group},
                    config=self.config,
                )
                self.envs[group_id] = group_wrapper
        else:
            # Create individual wrappers (original behavior)
            for tls_id in self.tls_ids:
                env_wrapper = _SingleTLSEnvironmentWrapper(
                    shared_env=self.shared_env,
                    tls_id=tls_id,
                    incoming_lanes=self.tls_lanes[tls_id][0],
                    outgoing_lanes=self.tls_lanes[tls_id][1],
                    config=self.config,
                )
                self.envs[tls_id] = env_wrapper

    def reset(self, *, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Reset all environments and return observations for each agent."""
        # Reset shared environment (this starts TraCI)
        obs, _ = self.shared_env.reset(seed=seed)
        
        # Get observations for each agent (group or individual TLS)
        observations = {}
        for agent_id, env in self.envs.items():
            observations[agent_id] = env.get_observation()
        
        return observations

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict]:
        """
        Step all environments with actions from all agents.

        Args:
            actions: Dictionary mapping agent ID (TLS ID or group ID) to action (flattened integer)

        Returns:
            Tuple of (observations, rewards, dones, infos) for each agent
        """
        # Apply actions from all agents
        for agent_id, action in actions.items():
            if agent_id in self.envs:
                self.envs[agent_id].apply_action(action)
        
        # Step the shared simulation
        obs, reward, terminated, truncated, info = self.shared_env.step(0)  # Dummy action, already applied above
        
        # Collect observations, rewards, and info for each agent
        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        
        for agent_id, env in self.envs.items():
            observations[agent_id] = env.get_observation()
            rewards[agent_id] = env.get_reward()
            dones[agent_id] = terminated or truncated
            infos[agent_id] = env.get_info()
        
        return observations, rewards, dones, infos

    def close(self) -> None:
        """Close the shared environment."""
        self.shared_env.close()

    def get_action_spaces(self) -> Dict[str, gym.Space]:
        """Get action space for each agent."""
        return {agent_id: env.action_space for agent_id, env in self.envs.items()}

    def get_observation_spaces(self) -> Dict[str, gym.Space]:
        """Get observation space for each agent."""
        return {agent_id: env.observation_space for agent_id, env in self.envs.items()}


class _SingleTLSEnvironmentWrapper:
    """Wrapper for a single TLS within a shared SUMO simulation."""

    def __init__(
        self,
        shared_env: SUMOEnvironment,
        tls_id: str,
        incoming_lanes: List[str],
        outgoing_lanes: List[str],
        config: SUMOEnvironmentConfig,
    ) -> None:
        self.shared_env = shared_env
        self.tls_id = tls_id
        self.incoming_lanes = tuple(incoming_lanes)
        self.outgoing_lanes = tuple(outgoing_lanes)
        self.config = config
        
        # Initialize phase map for this TLS
        self._phase_map: Tuple[str, ...] = ("GGrr", "rrGG")  # Default
        self._current_phase: int = 0
        self._current_phase_duration: int = 0
        self._last_action: int = 0
        
        # Initialize phase map
        self._initialize_phase_map()
        
        # Set up action and observation spaces
        features_per_lane = 3
        if self.config.enable_duration_control:
            num_durations = len(self.config.duration_options)
            self.action_space = gym.spaces.MultiDiscrete([len(self._phase_map), num_durations])
        else:
            self.action_space = gym.spaces.Discrete(len(self._phase_map))
        
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.incoming_lanes) * features_per_lane,),
            dtype=np.float32,
        )
        
        self._prev_total_queues: int = 0

    def _initialize_phase_map(self) -> None:
        """Initialize phase map for this TLS."""
        import traci
        
        try:
            current_state = traci.trafficlight.getRedYellowGreenState(self.tls_id)
            
            if self.config.extract_phases_from_sumo:
                try:
                    tls_program = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)
                    if tls_program and len(tls_program) > 0:
                        valid_phases = []
                        for phase in tls_program[0].phases:
                            state = phase.state
                            if 'G' in state and state.count('G') >= 2:
                                valid_phases.append(state)
                        
                        if valid_phases:
                            self._phase_map = tuple(valid_phases)
                        else:
                            self._generate_phase_map(current_state)
                    else:
                        self._generate_phase_map(current_state)
                except Exception:
                    self._generate_phase_map(current_state)
            else:
                self._generate_phase_map(current_state)
        except Exception:
            # Fallback to default
            self._phase_map = ("GGrr", "rrGG")

    def _generate_phase_map(self, current_state: str) -> None:
        """Generate phase map based on number of links."""
        num_links = len(current_state)
        
        if num_links == 4:
            self._phase_map = ("GGrr", "rrGG")
        elif num_links >= 2:
            half = num_links // 2
            phase1 = "G" * half + "r" * (num_links - half)
            phase2 = "r" * half + "G" * (num_links - half)
            self._phase_map = (phase1, phase2)
        else:
            self._phase_map = (current_state,)

    def apply_action(self, action: int) -> None:
        """Apply action to this TLS."""
        import traci
        
        if self.config.enable_duration_control:
            num_durations = len(self.config.duration_options)
            phase_idx = action // num_durations
            duration_idx = action % num_durations
            
            duration_steps = int(self.config.duration_options[duration_idx] / self.config.step_length)
            
            if phase_idx != self._current_phase or self._current_phase_duration <= 0:
                if self._current_phase_duration > 0 and self._current_phase_duration < int(self.config.min_green_time / self.config.step_length):
                    self._current_phase_duration = int(self.config.min_green_time / self.config.step_length)
                else:
                    state = self._phase_map[phase_idx]
                    traci.trafficlight.setRedYellowGreenState(self.tls_id, state)
                    self._current_phase = phase_idx
                    self._current_phase_duration = duration_steps
                    self._last_action = action
            else:
                self._current_phase_duration -= 1
        else:
            phase_idx = int(action)
            if phase_idx != self._current_phase:
                state = self._phase_map[phase_idx]
                traci.trafficlight.setRedYellowGreenState(self.tls_id, state)
                self._current_phase = phase_idx
                self._last_action = action

    def get_observation(self) -> np.ndarray:
        """Get observation for this TLS."""
        import traci
        
        queues = [int(traci.lane.getLastStepHaltingNumber(lane)) for lane in self.incoming_lanes]
        speeds = [float(traci.lane.getLastStepMeanSpeed(lane)) for lane in self.incoming_lanes]
        waits = [float(traci.lane.getWaitingTime(lane)) for lane in self.incoming_lanes]
        
        features: List[float] = []
        for queue, speed, wait in zip(queues, speeds, waits):
            queue_norm = min(queue / max(self.config.queue_capacity, 1e-6), 1.0)
            speed_norm = min(speed / max(self.config.max_speed, 1e-6), 1.0)
            wait_norm = min(wait / max(self.config.max_wait_time, 1e-6), 1.0)
            features.extend([queue_norm, speed_norm, wait_norm])
        
        return np.asarray(features, dtype=np.float32)

    def get_reward(self) -> float:
        """Get reward for this TLS."""
        import traci
        
        queues = [int(traci.lane.getLastStepHaltingNumber(lane)) for lane in self.incoming_lanes]
        waits = [float(traci.lane.getWaitingTime(lane)) for lane in self.incoming_lanes]
        
        current_total_queues = sum(queues)
        queue_penalty = self.config.queue_penalty * current_total_queues
        wait_penalty = self.config.wait_penalty * sum(waits)
        reward = -(queue_penalty + wait_penalty)
        
        if queue_penalty == 0:
            reward += self.config.clear_bonus
        
        queue_reduction = self._prev_total_queues - current_total_queues
        if queue_reduction > 0:
            reward += 0.1 * queue_reduction
        
        self._prev_total_queues = current_total_queues
        
        return float(reward)

    def get_info(self) -> Dict:
        """Get info for this TLS."""
        import traci
        
        queues = [int(traci.lane.getLastStepHaltingNumber(lane)) for lane in self.incoming_lanes]
        waits = [float(traci.lane.getWaitingTime(lane)) for lane in self.incoming_lanes]
        
        info = {
            "tls_id": self.tls_id,
            "action": self._last_action,
            "queues": queues,
            "waiting_times": waits,
            "current_phase": self._current_phase,
        }
        
        if self.config.enable_duration_control:
            info["phase_duration_remaining"] = self._current_phase_duration
        
        return info


class _GroupedTLSEnvironmentWrapper:
    """Wrapper for a group of TLS controlled by a single agent."""

    def __init__(
        self,
        shared_env: SUMOEnvironment,
        tls_ids: List[str],
        tls_lanes: Dict[str, Tuple[List[str], List[str]]],
        config: SUMOEnvironmentConfig,
    ) -> None:
        self.shared_env = shared_env
        self.tls_ids = tls_ids
        self.tls_lanes = tls_lanes
        self.config = config
        
        # Initialize phase maps and states for each TLS
        self.tls_phase_maps: Dict[str, Tuple[str, ...]] = {}
        self.tls_current_phases: Dict[str, int] = {}
        self.tls_phase_durations: Dict[str, int] = {}
        self.tls_last_actions: Dict[str, int] = {}
        self.tls_prev_queues: Dict[str, int] = {}
        
        # Initialize phase maps for all TLS in group
        self._initialize_phase_maps()
        
        # Set up action space: MultiDiscrete with one action per TLS
        if self.config.enable_duration_control:
            action_nvecs = []
            for tls_id in self.tls_ids:
                num_phases = len(self.tls_phase_maps[tls_id])
                num_durations = len(self.config.duration_options)
                action_nvecs.extend([num_phases, num_durations])
            self.action_space = gym.spaces.MultiDiscrete(action_nvecs)
        else:
            action_nvecs = [len(self.tls_phase_maps[tls_id]) for tls_id in self.tls_ids]
            self.action_space = gym.spaces.MultiDiscrete(action_nvecs)
        
        # Observation space: concatenation of all TLS observations
        total_obs_size = sum(
            len(tls_lanes[tls_id][0]) * 3  # 3 features per lane
            for tls_id in self.tls_ids
        )
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(total_obs_size,),
            dtype=np.float32,
        )

    def _initialize_phase_maps(self) -> None:
        """Initialize phase maps for all TLS in the group."""
        import traci
        
        # Start TraCI connection temporarily if needed
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
                    
                except Exception:
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

    def apply_action(self, action: int) -> None:
        """Apply action to all TLS in the group."""
        import traci
        
        # Decode flattened action to MultiDiscrete
        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            # Decode action from flattened integer
            action_array = np.zeros(len(self.action_space.nvec), dtype=np.int64)
            remaining = action
            
            if self.config.enable_duration_control:
                # Each TLS has [phase, duration] pair
                # Decode from right to left (least significant first)
                for i in range(len(self.tls_ids) - 1, -1, -1):
                    tls_id = self.tls_ids[i]
                    num_durations = len(self.config.duration_options)
                    num_phases = len(self.tls_phase_maps[tls_id])
                    
                    duration_idx = remaining % num_durations
                    remaining = remaining // num_durations
                    phase_idx = remaining % num_phases
                    remaining = remaining // num_phases
                    
                    action_array[i * 2] = phase_idx
                    action_array[i * 2 + 1] = duration_idx
                    
                    # Apply to TLS
                    duration_steps = int(self.config.duration_options[duration_idx] / self.config.step_length)
                    
                    if phase_idx != self.tls_current_phases[tls_id] or self.tls_phase_durations[tls_id] <= 0:
                        if self.tls_phase_durations[tls_id] > 0 and self.tls_phase_durations[tls_id] < int(self.config.min_green_time / self.config.step_length):
                            self.tls_phase_durations[tls_id] = int(self.config.min_green_time / self.config.step_length)
                        else:
                            state = self.tls_phase_maps[tls_id][phase_idx]
                            traci.trafficlight.setRedYellowGreenState(tls_id, state)
                            self.tls_current_phases[tls_id] = phase_idx
                            self.tls_phase_durations[tls_id] = duration_steps
                            self.tls_last_actions[tls_id] = phase_idx * num_durations + duration_idx
                    else:
                        self.tls_phase_durations[tls_id] -= 1
            else:
                # Each TLS has one phase action
                # Decode from right to left
                for i in range(len(self.tls_ids) - 1, -1, -1):
                    tls_id = self.tls_ids[i]
                    num_phases = len(self.tls_phase_maps[tls_id])
                    phase_idx = remaining % num_phases
                    remaining = remaining // num_phases
                    
                    action_array[i] = phase_idx
                    
                    # Apply to TLS
                    if phase_idx != self.tls_current_phases[tls_id]:
                        state = self.tls_phase_maps[tls_id][phase_idx]
                        traci.trafficlight.setRedYellowGreenState(tls_id, state)
                        self.tls_current_phases[tls_id] = phase_idx
                        self.tls_last_actions[tls_id] = phase_idx

    def get_observation(self) -> np.ndarray:
        """Get concatenated observation from all TLS in the group."""
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

    def get_reward(self) -> float:
        """Get combined reward from all TLS in the group."""
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

    def get_info(self) -> Dict:
        """Get info for all TLS in the group."""
        import traci
        
        info = {
            "tls_ids": self.tls_ids,
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

