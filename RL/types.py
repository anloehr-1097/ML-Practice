from gymnasium.core import ObsType, ActType
from typing import TypeVar, Tuple

StateAndOrAction = TypeVar("StateAndOrAction", ObsType, Tuple[ObsType, ActType])
