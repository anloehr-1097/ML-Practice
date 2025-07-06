import gymnasium as gym
import unittest
from RL.value_function import (
    StateValueFunctionFactory,
    DiscreteStateValueFunction,
    StateActionValueFunctionFactory,
    DiscreteStateActionValueFunction,
)
from RL.simulate import StateActionTemporalDifferenceWrapper
from RL.policy import GreedyPolicy


def test_sim() -> None:
    cliff_walk: gym.Env = gym.make("CliffWalking-v0", render_mode="human")
    valfunc: DiscreteStateActionValueFunction = StateActionValueFunctionFactory.create(
        cliff_walk
    )
    greedy_cliff_walk: GreedyPolicy = GreedyPolicy(valfunc)

    sim = StateActionTemporalDifferenceWrapper(cliff_walk, valfunc, greedy_cliff_walk)
    return sim


if __name__ == "__main__":
    sim = test_sim()
    sim.run(1000)
