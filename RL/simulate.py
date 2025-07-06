import gymnasium as gym
from typing import Callable, Dict, Any, Optional, Tuple
from gymnasium.core import ObsType, ActType
from .value_function import ValueFunction, ValueFunction
from .algorithms import (
    state_temporal_difference_update,
    state_action_temporal_difference_update,
)
from .policy import Policy, GreedyPolicy
from .utils import print_all


class Simulation:
    def __init__(self, env: gym.Env, policy: Optional[Policy] = None):
        self.env: gym.Env = env
        self.policy: Optional[Policy] = policy if policy else None
        self.total_reward: float = 0
        self.obs: ObsType
        self.obs_prime: ObsType
        self.action: ActType
        self.info: Dict

        self.reset()

    def get_action(self, obs: ObsType) -> ActType:
        return self.policy(obs).item() if self.policy else self.env.action_space.sample()  # type: ignore

    def run(self, episodes: int = 10000):
        episode: int = 0
        while episode < episodes:
            print(f"Episode {episode}")
            action: ActType = self.get_action(self.obs_prime)  # type: ignore
            terminated, truncated = self.step(action)
            episode += 1
            if terminated:
                print(f"Termineated at epoch {episode}.")
                self.reset()
            if truncated:
                print(f"Truncated at epoch {episode}.")
                self.reset()

    def reset(self):
        self.obs_prime, self.info = self.env.reset()

    # modify these in wrapper, for example with temporal_difference
    # def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict]:
    def step(self, action: ActType) -> Tuple[bool, bool]:
        self.obs = self.obs_prime
        self.obs_prime, reward, terminated, truncated, self.info = self.env.step(action)  # type: ignore
        print_all(self.obs, action, reward, self.obs_prime)
        return terminated, truncated


class StateTemporalDifferenceWrapper(Simulation):
    alpha: float = 0.7
    gamma: float = 0.7

    # overwrite the step method to do temporal difference update
    def __init__(
        self,
        env: gym.Env,
        val_fun: ValueFunction,
        policy: Optional[Policy] = None,
    ):
        super().__init__(env, policy)
        self.value_function: ValueFunction = val_fun

    def step(self, action: ActType) -> Tuple[bool, bool]:
        self.obs = self.obs_prime
        self.obs_prime, reward, terminated, truncated, self.info = self.env.step(action)  # type: ignore
        state_temporal_difference_update(
            self.value_function,
            self.obs,
            float(reward),
            self.obs_prime,
            self.alpha,
            self.gamma,
        )

        print_all(self.obs, action, reward, self.obs_prime)
        return terminated, truncated


class StateActionTemporalDifferenceWrapper(Simulation):
    alpha: float = 0.7
    gamma: float = 0.7

    # overwrite the step method to do temporal difference update
    def __init__(
        self,
        env: gym.Env,
        val_fun: ValueFunction,
        policy: Optional[Policy] = None,
    ):
        super().__init__(env, policy)
        self.value_function: ValueFunction = val_fun
        self.action: Optional[ActType] = None
        self.next_action: Optional[ActType] = None

        print(f"Val FUn pre temp diff: {val_fun}")
        print(f"Val Fun pre temp diff at 1, 1: {val_fun((1,1))}")

        print(f"Val FUn pre temp diff: {self.value_function}")
        print(f"Val Fun pre temp diff at 1, 1: {self.value_function((1,1))}")

    def step(self, action: ActType) -> Tuple[bool, bool]:
        if self.next_action:
            self.action = self.next_action
        else:
            self.action = action  # type: ignore

        self.obs = self.obs_prime
        self.obs_prime, reward, terminated, truncated, self.info = self.env.step(self.action)  # type: ignore
        self.next_action = self.get_action(self.obs_prime)
        print(f"Val FUn pre temp diff: {self.value_function}")
        print(f"Val Fun pre temp diff at 1, 1: {self.value_function((1,1))}")

        state_action_temporal_difference_update(
            self.value_function,
            self.obs,
            self.action,
            float(reward),
            self.obs_prime,
            self.next_action,
            self.alpha,
            self.gamma,
        )

        print_all(self.obs, action, reward, self.obs_prime)
        return terminated, truncated
