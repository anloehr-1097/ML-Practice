from typing import Tuple
from gymnasium.core import ObsType, ActType
import torch
import numpy as np

from .value_function import (
    ValueFunction,
    ContinuousStateValueFunction,
    DiscreteStateValueFunction,
    DiscreteStateActionValueFunction,
)

from .types import StateAndOrAction


def temporal_difference_update(
    val_function: ValueFunction,
    cur: StateAndOrAction,
    nxt: StateAndOrAction,
    ret: float,
    alpha: float,
    gamma: float,
) -> None:

    # assert that types match
    assert (
        isinstance(cur, Tuple)
        and isinstance(val_function, DiscreteStateActionValueFunction)
        or (
            isinstance(cur, int)
            and isinstance(val_function, DiscreteStateValueFunction)
        )
        or (
            isinstance(cur, np.ndarray)
            and isinstance(val_function, ContinuousStateValueFunction)
        )
    ), "Types do not match."

    td_error: torch.Tensor = ret + gamma * val_function(nxt) - val_function(cur)

    if isinstance(val_function, ContinuousStateValueFunction):
        loss: torch.Tensor = -td_error
        print(loss)
        raise NotImplementedError
        # gradient descent

    elif isinstance(val_function, DiscreteStateActionValueFunction):
        val_function.update(cur, float(alpha * td_error))

    return None


def state_temporal_difference_update(
    val_function: ValueFunction,
    obs: ObsType,
    ret: float,
    next_obs: ObsType,
    alpha: float,
    gamma: float,
) -> None:

    td_error: torch.Tensor = ret + gamma * val_function(next_obs) - val_function(obs)
    print(f"td_error: {td_error}")

    if isinstance(val_function, ContinuousStateValueFunction):
        loss: torch.Tensor = -td_error
        print(loss)
        raise NotImplementedError(
            "Implement td update for continuous state value functions"
        )
        # gradient descent

    elif isinstance(val_function, DiscreteStateValueFunction):
        val_function.update(obs, alpha * td_error)

    return None


def state_action_temporal_difference_update(
    val_function: ValueFunction,
    obs: ObsType,
    action: ActType,
    ret: float,
    next_obs: ObsType,
    next_action: ActType,
    alpha: float,
    gamma: float,
) -> None:
    print(val_function)

    td_error: torch.Tensor = (
        ret
        + gamma * val_function((next_obs, next_action))
        - val_function((obs, action))
    )

    if isinstance(val_function, ContinuousStateValueFunction):
        loss: torch.Tensor = -td_error
        print(loss)
        raise NotImplementedError
        # gradient descent

    elif isinstance(val_function, DiscreteStateActionValueFunction):
        val_function.update((obs, action), float(alpha * td_error))

    return None
