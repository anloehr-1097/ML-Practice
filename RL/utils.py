from typing import List, Tuple, Dict

from gymnasium.core import ObsType, ActType


def print_all(obs, action, reward, next_obs) -> None:
    print(
        f"Observation: {obs}\tAction: {action}\treward: {reward}\tNext Observation: {next_obs}"
    )
    return None


def first_visits(
    trajectory: List[Tuple[ObsType, ActType, float]],
) -> Dict[Tuple[ObsType, ActType], int]:
    visited = dict()
    for idx, (obs, action, _) in enumerate(trajectory):
        if (obs, action) not in visited:
            visited[(obs, action)] = idx
    return visited
