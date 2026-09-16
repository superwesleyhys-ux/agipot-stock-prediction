"""Small local contracts for the extracted, offline scoring components.

These dataclasses deliberately contain no service, persistence or account state.
Use :func:`score_stock` for validated daily-history inputs. Advanced callers may
construct a ``SkillContext`` to compose the individual scoring components.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def validate_finite_tree(value: Any, path: str = "payload") -> None:
    """Reject unsupported values and non-finite numbers at component boundaries."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: keys must be strings")
            validate_finite_tree(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            validate_finite_tree(child, f"{path}[{index}]")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path}: number must be finite")
    elif value is not None and not isinstance(value, (str, int, bool)):
        raise ValueError(f"{path}: unsupported value {type(value).__name__}")


@dataclass(frozen=True)
class HistoryBar:
    timestamp: datetime
    close: float
    volume: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("history timestamp must have a timezone")
        if not math.isfinite(self.close) or self.close <= 0:
            raise ValueError("close must be finite and positive")
        if not math.isfinite(self.volume) or self.volume < 0:
            raise ValueError("volume must be finite and non-negative")
        if not math.isfinite(self.close * self.volume):
            raise ValueError("close multiplied by volume must be finite")


@dataclass(frozen=True)
class MarketDataSnapshot:
    as_of: datetime
    universe: tuple[str, ...] = ()
    histories: Mapping[str, tuple[HistoryBar, ...]] = field(default_factory=dict)
    external_features: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise ValueError("as_of must have a timezone")
        validate_finite_tree(self.external_features, "external_features")


@dataclass(frozen=True)
class SkillVersion:
    implementation_key: str = "standalone"
    version: str = "1"


@dataclass(frozen=True)
class SkillBinding:
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillContext:
    snapshot: MarketDataSnapshot
    state: Mapping[str, Any] = field(default_factory=dict)
    binding: SkillBinding = field(default_factory=SkillBinding)

    def __post_init__(self) -> None:
        validate_finite_tree(self.state, "state")
        validate_finite_tree(self.binding.config, "binding.config")


@dataclass(frozen=True)
class RawSkillOutput:
    result_type: str
    payload: Mapping[str, Any]
    confidence: float | None = None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_finite_tree(self.payload)
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1
        ):
            raise ValueError("skill confidence must be finite and in [0, 1]")
