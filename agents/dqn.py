"""Deep Q-Network agent implementation."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base import AgentTransition, BaseAgent


@dataclass(slots=True)
class DQNConfig:
    """Configuration for the DQN agent."""

    gamma: float = 0.99
    learning_rate: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    batch_size: int = 64
    replay_size: int = 50_000
    target_update_interval: int = 200
    min_replay_size: int = 1_000
    hidden_layers: Iterable[int] = (128, 128)
    device: str = "cpu"


class QNetwork(nn.Module):
    """Simple multilayer perceptron for approximating Q-values."""

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: Iterable[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)


class DQNAgent(BaseAgent):
    """Vanilla DQN agent with target network and replay buffer."""

    def __init__(self, state_size: int, action_size: int, config: Optional[DQNConfig] = None) -> None:
        super().__init__(action_size)
        self.state_size = state_size
        self.config = config or DQNConfig()

        self.device = torch.device(self.config.device)
        self.policy_net = QNetwork(state_size, action_size, self.config.hidden_layers).to(self.device)
        self.target_net = QNetwork(state_size, action_size, self.config.hidden_layers).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        self.criteria = nn.MSELoss()

        self.memory: Deque[AgentTransition] = deque(maxlen=self.config.replay_size)
        self.epsilon = self.config.epsilon_start
        self._updates = 0

        self.sync_target_network(hard=True)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------
    def select_action(self, observation: np.ndarray, explore: bool = True) -> int:  # type: ignore[override]
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(obs_tensor)
        action = int(torch.argmax(q_values, dim=1).item())
        return action

    def observe(self, transition: AgentTransition) -> None:  # type: ignore[override]
        self.memory.append(transition)

    def update(self) -> Dict[str, float]:  # type: ignore[override]
        if len(self.memory) < max(self.config.min_replay_size, self.config.batch_size):
            return {}

        batch = random.sample(self.memory, self.config.batch_size)
        batch_states = torch.as_tensor(
            np.stack([tr.state for tr in batch]), dtype=torch.float32, device=self.device
        )
        batch_actions = torch.as_tensor([tr.action for tr in batch], dtype=torch.int64, device=self.device)
        batch_rewards = torch.as_tensor([tr.reward for tr in batch], dtype=torch.float32, device=self.device)
        batch_next_states = torch.as_tensor(
            np.stack([tr.next_state for tr in batch]), dtype=torch.float32, device=self.device
        )
        batch_dones = torch.as_tensor([tr.done for tr in batch], dtype=torch.bool, device=self.device)

        current_q = self.policy_net(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(batch_next_states).max(1)[0]
            target_q = batch_rewards + self.config.gamma * next_q * (~batch_dones)

        loss = self.criteria(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._updates += 1
        if self._updates % self.config.target_update_interval == 0:
            self.sync_target_network()

        if self.epsilon > self.config.epsilon_end:
            self.epsilon *= self.config.epsilon_decay
            self.epsilon = max(self.epsilon, self.config.epsilon_end)

        return {"loss": float(loss.item()), "epsilon": float(self.epsilon)}

    def reset(self) -> None:  # type: ignore[override]
        pass

    def save(self, target: Path) -> None:  # type: ignore[override]
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy_net.state_dict(),
                "target": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "config": self.config,
                "state_size": self.state_size,
                "action_size": self.action_size,
            },
            target,
        )

    def load(self, source: Path) -> None:  # type: ignore[override]
        checkpoint = torch.load(source, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy"])
        self.target_net.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = float(checkpoint.get("epsilon", self.config.epsilon_end))

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def sync_target_network(self, hard: bool = False) -> None:
        if hard:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(policy_param.data)

