from dataclasses import dataclass


@dataclass
class Feature:
    metric: str
    description: str
    priority: int = 0
