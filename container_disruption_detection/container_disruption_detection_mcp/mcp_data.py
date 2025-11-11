from __future__ import annotations
import logging
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field

logger = logging.getLogger("container_disruption_detection_data")


class RootCauseResult(BaseModel):
    metric: str
    labels: Dict[str, Union[str, int, float]] = Field(default_factory=dict)
    score: float

class AnomalyInfo(BaseModel):
    machine_id: str = Field(default="", description="机器ID（默认空字符串）")
    metric: str = Field(default="", description="特征量名称（默认空字符串）")
    labels: Dict[str, Union[str, int, float]] = Field(default_factory=dict, description="劣化详细信息（默认空字典）")
    score: float = Field(default=0.0, description="事件异常分数（默认0.0）")
    entity_name: Optional[str] = Field(default="", description="gala-gopher上报对应的entity_name（默认空字符串）")
    details: Dict[str, Union[str, int, float, dict]] = Field(default_factory=dict,description="异常事件详细信息（默认空字典）")

class AnomalyResult(BaseModel):
    is_anomaly: bool = Field(default=False, description="是否为异常事件（默认False）")
    anomaly_info: List[AnomalyInfo] = Field(default_factory=list, description="异常事件详细信息列表（默认空列表）")


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
