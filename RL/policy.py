from .value_function import (
    DiscreteStateActionValueFunction,
    ContinuousStateValueFunction,
)
from gymnasium.core import ObsType, ActType
from abc import abstractmethod
import torch


class Policy:
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def sample(self, observation: ObsType) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, observation: ObsType) -> torch.Tensor:
        raise NotImplementedError


class GreedyPolicy(Policy):
    def __init__(self, value_func: DiscreteStateActionValueFunction):
        self.value_function = value_func

    def sample(self, observation: ObsType) -> torch.Tensor:
        return torch.argmax(self.value_function.action_values(observation))

    def __call__(self, observation: ObsType) -> torch.Tensor:
        return torch.argmax(self.value_function.action_values(observation))
