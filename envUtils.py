import gymnasium as gym
import numpy as np
from typing import Dict, Any

def make_frozenlake_env(map_name: str = "4x4", is_slippery: bool = True, seed: int = 777):
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)
    # Gymnasium seed pattern:
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def is_terminal(done: bool, truncated: bool) -> bool:
    return done or truncated

def success_from_info(info: Dict[str, Any]) -> bool:
    # En FrozenLake, recompensa 1 usualmente implica éxito, pero validamos por 'is_success' si existe.
    if 'is_success' in info:
        return bool(info['is_success'])
    return False

def moving_average(data, window: int = 50):
    if len(data) == 0:
        return np.array([])
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='same')
