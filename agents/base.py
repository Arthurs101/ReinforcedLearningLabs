"""Abstract base classes and data structures for RL agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np


@dataclass(slots=True)
class AgentTransition:
    """Single transition tuple used for experience replay."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class BaseAgent(ABC):
    """Defines the common interface shared by all RL agents."""

    def __init__(self, action_size: int) -> None:
        self.action_size = action_size

    @abstractmethod
    def select_action(self, observation: np.ndarray, explore: bool = True) -> int:
        """Pick an action given the current observation."""

    def observe(self, transition: AgentTransition) -> None:
        """Record a transition for later learning."""

    def update(self) -> Dict[str, Any]:
        """Run a single optimisation step. Returns diagnostic metrics."""

    def reset(self) -> None:
        """Reset any episodic state (optional)."""

    def save(self, target: Path) -> None:
        """Persist agent parameters to disk."""

    def load(self, source: Path) -> None:
        """Load agent parameters from disk."""

