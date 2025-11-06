"""Gymnasium-compatible environment wrapper around a SUMO simulation."""

from __future__ import annotations

import os
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

        self._phase_map: Tuple[str, ...] = ("GGrr", "rrGG")
        self._tls_id = self.artifacts.tls_id
        self._incoming_lanes = tuple(self.artifacts.incoming_lanes)

        features_per_lane = 3  # queue, speed, waiting time
        self.action_space = gym.spaces.Discrete(len(self._phase_map))
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self._incoming_lanes) * features_per_lane,),
            dtype=np.float32,
        )

        self._conn_active = False
        self._last_action = 0
        self._current_step = 0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):  # type: ignore[override]
        super().reset(seed=seed)

        self._start_traci(seed if seed is not None else self.config.seed)
        self._current_step = 0
        self._last_action = 0

        # Warm-up period to populate queues before control begins
        for _ in range(max(0, self.config.warmup_steps)):
            traci.simulationStep()

        observation = self._collect_observation()
        info = self._collect_info()
        return observation, info

    def step(self, action: int):  # type: ignore[override]
        action_idx = int(action)
        if action_idx < 0 or action_idx >= len(self._phase_map):
            raise ValueError(f"Action {action_idx} is out of bounds for phase map")

        if action_idx != self._last_action:
            self._apply_phase(action_idx)
            self._last_action = action_idx

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
        print(
            f"Step {self._current_step:04d} | Action {self._last_action} | "
            f"Queues {queues}"
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
        return {
            "step": self._current_step,
            "action": self._last_action,
            "queues": self._lane_queues(),
            "waiting_times": self._lane_wait_times(),
        }

