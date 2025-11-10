"""Gymnasium-compatible environment wrapper around a SUMO simulation."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np

try:  # pragma: no cover - import guard for environments without SUMO
    import traci  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "The 'traci' module is required. Please ensure SUMO is installed and available on PATH."
    ) from exc

from .scenario import ScenarioArtifacts, SimpleIntersectionScenario, OSMScenario


@dataclass(slots=True)
class SUMOEnvironmentConfig:
    """Configuration parameters for the SUMO environment wrapper."""

    max_steps: int = 900
    step_length: float = 1.0
    warmup_steps: int = 0
    frame_skip: int = 1
    use_gui: bool = False
    sumo_binary: str = "sumo"
    sumo_gui_binary: str = "sumo-gui"
    queue_capacity: float = 20.0
    max_speed: float = 13.89  # 50 km/h in m/s
    queue_penalty: float = 1.0
    wait_penalty: float = 0.05
    clear_bonus: float = 2.5
    seed: Optional[int] = None
    additional_sumo_args: Tuple[str, ...] = field(default_factory=tuple)
    # Phase duration control
    enable_duration_control: bool = True
    duration_options: Tuple[int, ...] = (5, 10, 15, 20)  # Duration options in seconds
    min_green_time: int = 5  # Minimum green time for safety
    # Phase extraction
    extract_phases_from_sumo: bool = True  # Extract phases from SUMO instead of generating

    def resolve_binary(self) -> str:
        """Pick the appropriate SUMO binary depending on GUI preference."""

        return self.sumo_gui_binary if self.use_gui else self.sumo_binary


class SUMOEnvironment(gym.Env[np.ndarray, int]):
    """Simple single-junction traffic-light control environment backed by SUMO."""

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        config: Optional[SUMOEnvironmentConfig] = None,
        scenario: Optional[Union[SimpleIntersectionScenario, OSMScenario]] = None,
    ) -> None:
        super().__init__()

        self.config = config or SUMOEnvironmentConfig()
        self.scenario_builder = scenario or SimpleIntersectionScenario()
        self.artifacts: ScenarioArtifacts = self.scenario_builder.build()

        self._tls_id = self.artifacts.tls_id
        self._incoming_lanes = tuple(self.artifacts.incoming_lanes)
        self._phase_map: Tuple[str, ...] = ("GGrr", "rrGG")  # Default, will be updated after TraCI starts
        self._num_links: Optional[int] = None  # Will be set after TraCI connection
        self._duration_options = self.config.duration_options

        features_per_lane = 3  # queue, speed, waiting time
        # Action space will be updated after phase map is determined
        self.action_space = gym.spaces.Discrete(2)  # Temporary, updated in _initialize_phase_map
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self._incoming_lanes) * features_per_lane,),
            dtype=np.float32,
        )

        self._conn_active = False
        self._last_action: Union[int, Tuple[int, int]] = 0  # Can be int or (phase, duration)
        self._current_step = 0
        self._current_phase_duration: int = 0  # Steps remaining in current phase
        self._current_phase: int = 0  # Current phase index

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):  # type: ignore[override]
        super().reset(seed=seed)

        self._start_traci(seed if seed is not None else self.config.seed)
        self._current_step = 0
        self._last_action = 0 if not self.config.enable_duration_control else (0, 0)
        self._current_phase_duration = 0
        self._current_phase = 0

        # Initialize phase map based on actual traffic light configuration
        self._initialize_phase_map()

        # Warm-up period to populate queues before control begins
        for _ in range(max(0, self.config.warmup_steps)):
            traci.simulationStep()

        observation = self._collect_observation()
        info = self._collect_info()
        return observation, info

    def step(self, action: Union[int, Tuple[int, int], np.ndarray]):  # type: ignore[override]
        """Execute one step in the environment.
        
        Args:
            action: If duration control is enabled, action is (phase_idx, duration_idx) or array [phase_idx, duration_idx].
                    Otherwise, action is phase_idx (int).
        """
        # Handle different action formats
        if self.config.enable_duration_control:
            if isinstance(action, (list, tuple, np.ndarray)):
                phase_idx = int(action[0])
                duration_idx = int(action[1])
            else:
                # Flattened action: decode from single integer
                num_durations = len(self._duration_options)
                phase_idx = int(action) // num_durations
                duration_idx = int(action) % num_durations
            
            if phase_idx < 0 or phase_idx >= len(self._phase_map):
                raise ValueError(f"Phase index {phase_idx} is out of bounds for phase map")
            if duration_idx < 0 or duration_idx >= len(self._duration_options):
                raise ValueError(f"Duration index {duration_idx} is out of bounds")
            
            duration_steps = int(self._duration_options[duration_idx] / self.config.step_length)
            
            # Check if we need to switch phase
            if phase_idx != self._current_phase or self._current_phase_duration <= 0:
                # Ensure minimum green time
                if self._current_phase_duration > 0 and self._current_phase_duration < int(self.config.min_green_time / self.config.step_length):
                    # Don't switch yet, extend current phase
                    self._current_phase_duration = int(self.config.min_green_time / self.config.step_length)
                else:
                    # Switch to new phase
                    self._apply_phase(phase_idx)
                    self._current_phase = phase_idx
                    self._current_phase_duration = duration_steps
                    self._last_action = (phase_idx, duration_idx)
            else:
                # Continue current phase, decrement duration
                self._current_phase_duration -= 1
        else:
            # Original behavior: phase selection only
            phase_idx = int(action)
            if phase_idx < 0 or phase_idx >= len(self._phase_map):
                raise ValueError(f"Action {phase_idx} is out of bounds for phase map")

            if phase_idx != self._current_phase:
                self._apply_phase(phase_idx)
                self._current_phase = phase_idx
                self._last_action = phase_idx

        for _ in range(max(1, self.config.frame_skip)):
            traci.simulationStep()

        self._current_step += 1

        observation = self._collect_observation()
        reward = self._calculate_reward()
        terminated = self._current_step >= self.config.max_steps
        truncated = False
        info = self._collect_info()

        # Automatically terminate when no vehicles remain
        if traci.simulation.getMinExpectedNumber() == 0:
            terminated = True

        return observation, reward, terminated, truncated, info

    def render(self):  # pragma: no cover - rendering is CLI-based
        queues = self._lane_queues()
        if self.config.enable_duration_control and isinstance(self._last_action, tuple):
            action_str = f"Phase {self._last_action[0]}, Duration {self._duration_options[self._last_action[1]]}s"
        else:
            action_str = f"Phase {self._last_action}"
        print(
            f"Step {self._current_step:04d} | Action {action_str} | "
            f"Queues {queues} | Phase duration remaining: {self._current_phase_duration}"
        )

    def close(self):
        if self._conn_active:
            traci.close(False)
            self._conn_active = False
        self.scenario_builder.cleanup()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _start_traci(self, seed: Optional[int]) -> None:
        if self._conn_active:
            traci.close(False)
            self._conn_active = False

        cmd = [
            self.config.resolve_binary(),
            "-c",
            os.fspath(self.artifacts.config_file),
            "--step-length",
            f"{self.config.step_length:.3f}",
            "--no-step-log",
            "true",
        ]

        if seed is not None:
            cmd.extend(["--seed", str(int(seed))])

        cmd.extend(self.config.additional_sumo_args)

        traci.start(cmd)
        self._conn_active = True

    def _initialize_phase_map(self) -> None:
        """Initialize phase map based on the actual traffic light configuration.
        
        If extract_phases_from_sumo is True, extracts valid phases from SUMO's traffic light definition.
        Otherwise, generates phases based on number of links.
        """
        try:
            # Get current state to determine number of controlled links
            current_state = traci.trafficlight.getRedYellowGreenState(self._tls_id)
            self._num_links = len(current_state)
            
            if self.config.extract_phases_from_sumo:
                # Try to extract phases from SUMO's traffic light definition
                try:
                    tls_program = traci.trafficlight.getCompleteRedYellowGreenDefinition(self._tls_id)
                    if tls_program and len(tls_program) > 0:
                        # Extract green phases (phases with at least 2 green lights)
                        valid_phases = []
                        for phase in tls_program[0].phases:
                            state = phase.state
                            # Include phases with green lights (skip pure red/yellow phases)
                            if 'G' in state and state.count('G') >= 2:
                                valid_phases.append(state)
                        
                        if valid_phases:
                            self._phase_map = tuple(valid_phases)
                            print(f"Extracted {len(valid_phases)} phases from SUMO: {self._phase_map}", file=sys.stderr)
                        else:
                            # Fallback to generated phases
                            self._generate_phase_map()
                    else:
                        # Fallback to generated phases
                        self._generate_phase_map()
                except Exception as e:
                    print(f"Warning: Could not extract phases from SUMO: {e}", file=sys.stderr)
                    print("Falling back to generated phases", file=sys.stderr)
                    self._generate_phase_map()
            else:
                # Use generated phases
                self._generate_phase_map()
            
            # Update action space based on whether duration control is enabled
            if self.config.enable_duration_control:
                # MultiDiscrete: [num_phases, num_durations]
                num_phases = len(self._phase_map)
                num_durations = len(self._duration_options)
                self.action_space = gym.spaces.MultiDiscrete([num_phases, num_durations])
            else:
                # Simple Discrete: phase selection only
                self.action_space = gym.spaces.Discrete(len(self._phase_map))
                
        except Exception as e:
            # Fallback to default if we can't get the state
            print(f"Warning: Could not determine traffic light phase configuration: {e}", file=sys.stderr)
            print(f"Using default phase map for 4 links", file=sys.stderr)
            self._phase_map = ("GGrr", "rrGG")
            if self.config.enable_duration_control:
                self.action_space = gym.spaces.MultiDiscrete([2, len(self._duration_options)])
            else:
                self.action_space = gym.spaces.Discrete(2)
    
    def _generate_phase_map(self) -> None:
        """Generate phase map based on number of links (fallback method)."""
        current_state = traci.trafficlight.getRedYellowGreenState(self._tls_id)
        self._num_links = len(current_state)
        
        if self._num_links == 4:
            # Simple 4-way intersection: alternate between two directions
            self._phase_map = ("GGrr", "rrGG")
        elif self._num_links >= 2:
            # For other configurations, create alternating phases
            # First half green, second half red, then vice versa
            half = self._num_links // 2
            phase1 = "G" * half + "r" * (self._num_links - half)
            phase2 = "r" * half + "G" * (self._num_links - half)
            self._phase_map = (phase1, phase2)
        else:
            # Fallback: use current state as single phase
            self._phase_map = (current_state,)

    def _apply_phase(self, action_idx: int) -> None:
        state = self._phase_map[action_idx]
        traci.trafficlight.setRedYellowGreenState(self._tls_id, state)

    def _lane_queues(self) -> List[int]:
        return [int(traci.lane.getLastStepHaltingNumber(lane)) for lane in self._incoming_lanes]

    def _lane_wait_times(self) -> List[float]:
        return [float(traci.lane.getWaitingTime(lane)) for lane in self._incoming_lanes]

    def _lane_speeds(self) -> List[float]:
        return [float(traci.lane.getLastStepMeanSpeed(lane)) for lane in self._incoming_lanes]

    def _collect_observation(self) -> np.ndarray:
        queues = self._lane_queues()
        speeds = self._lane_speeds()
        waits = self._lane_wait_times()

        features: List[float] = []
        for queue, speed, wait in zip(queues, speeds, waits):
            queue_norm = min(queue / max(self.config.queue_capacity, 1e-6), 1.0)
            speed_norm = min(speed / max(self.config.max_speed, 1e-6), 1.0)
            wait_norm = min(wait / max(self.config.max_steps * self.config.step_length, 1e-6), 1.0)
            features.extend([queue_norm, speed_norm, wait_norm])

        return np.asarray(features, dtype=np.float32)

    def _calculate_reward(self) -> float:
        queue_penalty = self.config.queue_penalty * sum(self._lane_queues())
        wait_penalty = self.config.wait_penalty * sum(self._lane_wait_times())
        reward = -(queue_penalty + wait_penalty)

        if queue_penalty == 0:
            reward += self.config.clear_bonus

        return float(reward)

    def _collect_info(self) -> Dict:
        info = {
            "step": self._current_step,
            "action": self._last_action,
            "queues": self._lane_queues(),
            "waiting_times": self._lane_wait_times(),
        }
        if self.config.enable_duration_control:
            info["current_phase"] = self._current_phase
            info["phase_duration_remaining"] = self._current_phase_duration
            if isinstance(self._last_action, tuple):
                info["phase_duration_seconds"] = self._duration_options[self._last_action[1]]
        return info

