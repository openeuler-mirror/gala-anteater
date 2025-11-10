from __future__ import annotations
import logging
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field

logger = logging.getLogger("container_disruption_detection_data")


class RootCauseModel(BaseModel):
    metric: str
    labels: Dict[str, Union[str, int, float]] = Field(default_factory=dict)
    score: float


class AnomalyModel(BaseModel):
    machine_id: str
    metric: str
    labels: Dict[str, Union[str, int, float]] = Field(default_factory=dict)
    score: float
    entity_name: Optional[str] = ""
    details: Dict[str, Union[str, int, float, dict]] = Field(default_factory=dict)
    root_causes: List[RootCauseModel] = Field(default_factory=list)


class KPIParam(BaseModel):
    metric: str
    entity_name: Optional[str] = ""
    params: Dict[str, Any] = Field(default_factory=dict)


class WindowParam(BaseModel):
    look_back: int = 20
    obs_size: int = 6


class ExtraConfig(BaseModel):
    extra_metrics: str = ""


class TSPoint(BaseModel):
    metric: str
    labels: Dict[str, Union[str, int, float]] = Field(default_factory=dict)
    time_stamps: List[int]
    values: List[float]


class TSPayload(BaseModel):
    items: List[TSPoint]


class RCARequest(BaseModel):
    victim: TSPoint
    context: TSPayload


class ReportType(str, Enum):
    normal = "normal"
    anomaly = "anomaly"
