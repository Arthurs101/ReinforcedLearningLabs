"""Improved DQN agent with Double DQN, Dueling architecture, and better training stability."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base import AgentTransition, BaseAgent


@dataclass(slots=True)
class ImprovedDQNConfig:
    """Configuration for the improved DQN agent."""

    gamma: float = 0.99
    learning_rate: float = 5e-4  # Lower learning rate for stability
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01  # Lower minimum for better exploration
    epsilon_decay_steps: int = 10_000  # Linear decay over steps instead of multiplicative
    batch_size: int = 64
    replay_size: int = 100_000  # Larger replay buffer
    target_update_interval: int = 100  # More frequent updates
    min_replay_size: int = 5_000  # More samples before training
    hidden_layers: Iterable[int] = (256, 256, 128)  # Deeper network
    device: str = "cpu"
    gradient_clip: float = 10.0  # Gradient clipping for stability
    use_huber_loss: bool = True  # More robust loss function
    use_double_dqn: bool = True  # Reduce overestimation bias
    use_dueling: bool = True  # Separate value and advantage


class DuelingQNetwork(nn.Module):
    """Dueling DQN architecture that separates state value and action advantages."""

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: Iterable[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        
        # Shared feature layers
        for hidden_dim in hidden_layers[:-1]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Value stream (how good is this state?)
        value_dim = hidden_layers[-1] if hidden_layers else prev_dim
        self.value_head = nn.Sequential(
            nn.Linear(prev_dim, value_dim),
            nn.ReLU(),
            nn.Linear(value_dim, 1)
        )
        
        # Advantage stream (how much better is each action?)
        self.advantage_head = nn.Sequential(
            nn.Linear(prev_dim, value_dim),
            nn.ReLU(),
            nn.Linear(value_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.shared_layers(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # This ensures identifiability and stability
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class ImprovedDQNAgent(BaseAgent):
    """Improved DQN agent with Double DQN, Dueling architecture, and better training stability."""

    def __init__(self, state_size: int, action_size: int, config: Optional[ImprovedDQNConfig] = None) -> None:
        super().__init__(action_size)
        self.state_size = state_size
        self.config = config or ImprovedDQNConfig()

        self.device = torch.device(self.config.device)
        
        # Use Dueling architecture if enabled, otherwise standard MLP
        if self.config.use_dueling:
            self.policy_net = DuelingQNetwork(
                state_size, action_size, self.config.hidden_layers
            ).to(self.device)
            self.target_net = DuelingQNetwork(
                state_size, action_size, self.config.hidden_layers
            ).to(self.device)
        else:
            # Standard MLP (from original DQN)
            layers: list[nn.Module] = []
            prev_dim = state_size
            for hidden_dim in self.config.hidden_layers:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, action_size))
            self.policy_net = nn.Sequential(*layers).to(self.device)
            self.target_net = nn.Sequential(*layers).to(self.device)
        
        # Initialize weights
        self._initialize_weights(self.policy_net)
        self._initialize_weights(self.target_net)
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        
        # Use Huber loss for robustness, MSE as fallback
        if self.config.use_huber_loss:
            self.criteria = nn.SmoothL1Loss()  # Huber loss
        else:
            self.criteria = nn.MSELoss()

        self.memory: Deque[AgentTransition] = deque(maxlen=self.config.replay_size)
        self.epsilon = self.config.epsilon_start
        self._updates = 0
        self._steps = 0  # Track total steps for epsilon decay

        self.sync_target_network(hard=True)

    def _initialize_weights(self, network: nn.Module) -> None:
        """Initialize network weights using Xavier uniform initialization."""
        for module in network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------
    def select_action(self, observation: np.ndarray, explore: bool = True) -> int:  # type: ignore[override]
        # Linear epsilon decay
        if explore:
            epsilon = max(
                self.config.epsilon_end,
                self.config.epsilon_start - (self.config.epsilon_start - self.config.epsilon_end) 
                * (self._steps / self.config.epsilon_decay_steps)
            )
            self.epsilon = epsilon
        else:
            epsilon = 0.0  # No exploration during evaluation
        
        if explore and random.random() < epsilon:
            return random.randrange(self.action_size)

        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(obs_tensor)
        action = int(torch.argmax(q_values, dim=1).item())
        return action

    def observe(self, transition: AgentTransition) -> None:  # type: ignore[override]
        self.memory.append(transition)
        self._steps += 1

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

        # Current Q-values
        current_q = self.policy_net(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            if self.config.use_double_dqn:
                # Double DQN: use policy network to select action, target network to evaluate
                next_actions = self.policy_net(batch_next_states).argmax(1)
                next_q = self.target_net(batch_next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q = self.target_net(batch_next_states).max(1)[0]
            
            target_q = batch_rewards + self.config.gamma * next_q * (~batch_dones)

        loss = self.criteria(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.gradient_clip)
        
        self.optimizer.step()

        self._updates += 1
        if self._updates % self.config.target_update_interval == 0:
            self.sync_target_network()

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
                "steps": self._steps,
                "updates": self._updates,
            },
            target,
        )

    def load(self, source: Path) -> None:  # type: ignore[override]
        # weights_only=False is safe here since we're loading our own checkpoints
        checkpoint = torch.load(source, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy"])
        self.target_net.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = float(checkpoint.get("epsilon", self.config.epsilon_end))
        self._steps = int(checkpoint.get("steps", 0))
        self._updates = int(checkpoint.get("updates", 0))

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def sync_target_network(self, hard: bool = False) -> None:
        if hard:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            # Soft update: copy parameters directly
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(policy_param.data)

