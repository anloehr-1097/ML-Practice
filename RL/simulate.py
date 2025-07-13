import gymnasium as gym
from typing import Dict, Optional, Tuple, List
from gymnasium.core import ObsType, ActType
from .value_function import ValueFunction, DiscreteStateActionValueFunction
from .algorithms import temporal_difference_update, state_temporal_difference_update
from .policy import Policy, EpsilonGreedyPolicy
from .utils import print_all, first_visits

import numpy as np


class Simulation:
    gamma: float = 0.9

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

    def run(self, num_episodes: int = 10000):
        episode: int = 0
        while episode < num_episodes:
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


class DiscreteMonteCarloSimulation(Simulation):
    def __init__(
        self, env: gym.Env, policy: Optional[Policy] = None, update_freq: int = 10
    ):
        assert isinstance(
            env.observation_space, gym.spaces.discrete.Discrete
        ), "This simulation only works with discrete observation spaces."
        assert isinstance(
            env.action_space, gym.spaces.discrete.Discrete
        ), "This simulation only works with discrete action spaces."
        super().__init__(env, policy)

        self.val_fun: DiscreteStateActionValueFunction = (
            DiscreteStateActionValueFunction(env)
        )
        self.trajectories: List[List[Tuple[ObsType, ActType, float]]] = []  # type: ignore
        self.current_trajectory: List[Tuple[ObsType, ActType, float]] = []  # type: ignore
        self.update_freq: int = update_freq  # how often to run policy improvement step

    def step(self, action: ActType) -> Tuple[bool, bool]:
        self.obs = self.obs_prime
        self.obs_prime, reward, terminated, truncated, self.info = self.env.step(action)
        self.current_trajectory.append((self.obs, action, float(reward)))

        print_all(self.obs, action, reward, self.obs_prime)
        return terminated, truncated

    def run(self, num_episodes: int = 10000):
        if self.update_freq / num_episodes > 0.1:
            print(
                f"Warning: update_freq ({self.update_freq}) is low compared to num_episodes ({num_episodes})."
            )
        episode: int = 0
        while episode < num_episodes:
            print(f"Episode {episode}")
            action: ActType = self.get_action(self.obs_prime)  # type: ignore
            terminated, truncated = self.step(action)
            if terminated or truncated:
                self.trajectories.append(self.current_trajectory)
                self.current_trajectory = []
                self.reset()
                if len(self.trajectories) % self.update_freq == 0:
                    print(f"Running policy improvement step at episode {episode}.")
                    self.policy_evaluation()
                    self.policy_improvement()

    def policy_improvement(self) -> None:
        """
        Implement the policy improvement step based on the collected trajectories.
        This is a placeholder method and should be implemented in subclasses.
        """

        self.policy = EpsilonGreedyPolicy(self.val_fun, epsilon=0.1)

    def policy_evaluation(self) -> None:
        """
        Implement the policy evaluation step based on the collected trajectories.
        This specifically is an implementation of first visit Monte Carlo method.
        """
        return_estimates: Dict[Tuple[int, int], float] = dict()
        counts: Dict[Tuple[int, int], int] = {
            (state, action): 0
            for state, action in zip(
                range(int(self.env.observation_space.n)),  # type: ignore
                range(int(self.env.action_space.n)),  # type: ignore
            )
        }

        for trajectory in self.trajectories:
            traj_first_visits: Dict[Tuple[ObsType, ActType], int] = first_visits(
                trajectory
            )
            # for each trajectory, calculate the return for each state-action pair in first-visit manner
            accumulated_return: float = 0
            discount_factors: np.ndarray = self.gamma ** np.arange(len(trajectory))
            idx: int = 0
            for obs, action, reward in reversed(trajectory):

                accumulated_return = accumulated_return * self.gamma + reward
                is_first_visit: bool = traj_first_visits.get((obs, action), -1) == (
                    len(trajectory) - idx
                )
                if is_first_visit:
                    return_estimates[(obs, action)] += accumulated_return
                    counts[(obs, action)] += 1
                idx += 1

        # average the return estimates
        val_fun: DiscreteStateActionValueFunction = DiscreteStateActionValueFunction(
            self.env
        )
        val_fun.zero()  # reset the value function
        for (state, action), total_return in return_estimates.items():
            if counts[(state, action)] > 0:
                val_fun.update((state, action), total_return / counts[(state, action)])
        self.val_fun = val_fun
        return None


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

        temporal_difference_update(
            self.value_function,
            (self.obs, self.action),
            (self.obs_prime, self.next_action),
            float(reward),
            self.alpha,
            self.gamma,
        )

        print_all(self.obs, action, reward, self.obs_prime)
        return terminated, truncated
