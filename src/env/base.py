"""Base environment abstraction defining the universal interface.

The universal closed-loop interface is:
    state = env.observe()
    action_distribution = policy(state)
    action = sample(action_distribution)
    next_state, reward, done, info = env.step(action)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional
import numpy as np


class Environment(ABC):
    """Abstract Base Class for all Pandu (pandu-jev) environments."""

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset environment to initial state and return initial observation."""
        raise NotImplementedError

    @abstractmethod
    def observe(self) -> Dict[str, Any]:
        """Return the current environment state / observation."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Apply action, transition to next state, and return (obs, reward, done, info)."""
        raise NotImplementedError

    @abstractmethod
    def get_feature_vector(self) -> np.ndarray:
        """Convert current state to fixed-length numeric vector suitable for policy input."""
        raise NotImplementedError

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Number of discrete actions or action dimension."""
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimension of feature vector returned by get_feature_vector()."""
        raise NotImplementedError

    @abstractmethod
    def render_ascii(self) -> str:
        """Return ASCII representation of current environment state."""
        raise NotImplementedError
