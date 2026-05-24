from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict

VALID_TASK_TYPES = {"binclass", "multiclass", "regression"}


@dataclass
class DatasetSpec:
    name: str
    task_type: str
    target: str
    column_names: List[str]
    numerical: List[str]
    categorical: List[str]
    dropped: List[str]
    default_naming: str
    source: Dict[str, str]
    notes: str = ""

    def __post_init__(self) -> None:
        if self.task_type not in VALID_TASK_TYPES:
            raise ValueError(f"{self.name}: bad task_type {self.task_type!r}")
        if self.default_naming not in {"real", "anonymized"}:
            raise ValueError(f"{self.name}: bad default_naming {self.default_naming!r}")
        if self.target in self.numerical or self.target in self.categorical:
            raise ValueError(f"{self.name}: target {self.target!r} must not be in feature lists")
        partition = set(self.numerical) | set(self.categorical) | {self.target}
        if partition != set(self.column_names):
            missing = set(self.column_names) - partition
            extra = partition - set(self.column_names)
            raise ValueError(f"{self.name}: partition mismatch missing={missing} extra={extra}")

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetSpec":
        return cls(
            name=d["name"], task_type=d["task_type"], target=d["target"],
            column_names=list(d["column_names"]), numerical=list(d["numerical"]),
            categorical=list(d["categorical"]), dropped=list(d.get("dropped", [])),
            default_naming=d.get("default_naming", "real"),
            source=dict(d["source"]), notes=d.get("notes", ""),
        )
