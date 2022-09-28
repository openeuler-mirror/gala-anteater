from dataclasses import dataclass


@dataclass
class Anomaly:
    metric: str = None
    labels: dict = None
    score: float = None
    entity_name: str = None
    description: str = None
