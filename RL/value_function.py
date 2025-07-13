from typing import Tuple
import torch
from torch import nn
from abc import abstractmethod
import gymnasium as gym
from gymnasium.core import ObsType, ActType
import numpy as np
from .types import StateAndOrAction


class ValueFunction:
    def __init__(self, env: gym.Env) -> None:
        self.env: gym.Env = env

    @abstractmethod
    def __call__(self, key: StateAndOrAction) -> torch.Tensor:
        raise NotImplementedError()


class DiscreteStateActionValueFunction(ValueFunction):
    # both state and actoin function discrete
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(
            env.observation_space, gym.spaces.discrete.Discrete
        ) and isinstance(env.action_space, gym.spaces.discrete.Discrete)
        state_space_size: int = int(env.observation_space.n)
        action_space_size: int = int(env.action_space.n)
        self.values: torch.Tensor = torch.rand((state_space_size, action_space_size))

    def __call__(self, key: Tuple[ObsType, ActType]):
        return self.values[int(key[0]), int(key[1])]  # type: ignore

    def action_values(self, key: ObsType) -> torch.Tensor:
        return self.values[int(key), :]  # type: ignore

    def update(self, key: Tuple[ObsType, ActType], value: float) -> None:
        self.values[int(key[0]), int(key[1])] += value  # type: ignore

    def zero(self) -> None:
        self.values.zero_()

    @classmethod
    def from_dict(cls, dct: dict, env: gym.Env) -> "DiscreteStateActionValueFunction":
        val_func: "DiscreteStateActionValueFunction" = cls(env)
        val_func.values = torch.zeros(
            (int(env.observation_space.n), int(env.action_space.n))
        )
        for k, v in dct.items():
            if not isinstance(k, tuple) or len(k) != 2:
                raise ValueError("Keys must be tuples of (state, action).")
            if not isinstance(v, float):
                raise ValueError("Values must be floats.")
            val_func.update(k, v)
        return val_func


class DiscreteStateValueFunction(ValueFunction):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.discrete.Discrete)
        state_space_size: int = int(env.observation_space.n)
        self.values: torch.Tensor = torch.rand(state_space_size)

    def __call__(self, obs: ObsType) -> torch.Tensor:
        return self.values[int(obs)]  # type: ignore

    def update(self, obs: ObsType, value) -> None:
        self.values[int(obs)] += value  # type: ignore


class ContinuousStateValueFunction(ValueFunction):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        # init value function approximation, simple neural net
        self.f1 = nn.Linear(np.size(np.size(env.observation_space.sample())), 20)
        self.f2 = nn.Linear(20, 10)
        self.out = nn.Linear(10, 1)

    def __call__(self, obs: ObsType) -> torch.Tensor:
        return self.out(self.f2(self.f1(torch.Tensor(obs))))


class StateValueFunctionFactory:
    @staticmethod
    def create(env: gym.Env) -> ValueFunction:
        if isinstance(env.observation_space, gym.spaces.box.Box):
            return ContinuousStateValueFunction(env)
        elif isinstance(env.observation_space, gym.spaces.discrete.Discrete):
            return DiscreteStateValueFunction(env)
        else:
            raise NotImplementedError("Not implemented")


class StateActionValueFunctionFactory:
    @staticmethod
    def create(env: gym.Env) -> ValueFunction:
        if isinstance(env.observation_space, gym.spaces.box.Box):
            return ContinuousStateValueFunction(env)
        elif isinstance(env.observation_space, gym.spaces.discrete.Discrete):
            return DiscreteStateActionValueFunction(env)
        else:
            raise NotImplementedError("Not implemented")
