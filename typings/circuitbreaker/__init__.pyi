from collections.abc import Callable
from typing import Any, TypeVar

_T = TypeVar("_T")

class CircuitBreakerError(Exception): ...

def circuit(*args: Any, **kwargs: Any) -> Callable[[Callable[..., _T]], Callable[..., _T]]: ...
