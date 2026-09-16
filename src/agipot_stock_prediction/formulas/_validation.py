"""Validate declared input types before comparisons can hide missing/NaN data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import wraps
import inspect
from types import UnionType
from typing import Any, Callable, TypeVar, Union, get_args, get_origin, get_type_hints

from agipot_stock_prediction.research.contracts import ResearchDataError, finite_number

F = TypeVar("F", bound=Callable[..., Any])


def _value(value: Any, annotation: Any, name: str) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (UnionType, Union):
        for candidate in args:
            try:
                return _value(value, candidate, name)
            except (ResearchDataError, TypeError):
                pass
        raise ResearchDataError(f"{name} does not match its declared input type")
    if annotation is float:
        return finite_number(value, name)
    if annotation is int:
        if type(value) is not int:
            raise ResearchDataError(f"{name} must be an integer")
        return value
    if annotation is bool:
        if type(value) is not bool:
            raise ResearchDataError(f"{name} must be a boolean")
        return value
    if origin in (Sequence, tuple):
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ResearchDataError(f"{name} must be a sequence")
        return tuple(_value(item, args[0], f"{name}[{index}]") for index, item in enumerate(value))
    if origin is Mapping:
        if not isinstance(value, Mapping):
            raise ResearchDataError(f"{name} must be a mapping")
        return {_value(key, args[0], name): _value(item, args[1], f"{name}[{key}]") for key, item in value.items()}
    if annotation is not Any and not isinstance(value, annotation):
        raise ResearchDataError(f"{name} does not match its declared input type")
    return value


def validated_inputs(function: F) -> F:
    """Enforce primitive/sequence/mapping annotations for public pure formulas."""
    signature = inspect.signature(function)
    annotations = get_type_hints(function)

    @wraps(function)
    def checked(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, value in tuple(bound.arguments.items()):
            bound.arguments[name] = _value(value, annotations[name], f"{function.__name__}.{name}")
        return function(*bound.args, **bound.kwargs)

    return checked
