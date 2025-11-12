from __future__ import annotations
import logging
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field

logger = logging.getLogger("container_disruption_detection_data")


# --------------------------- 基础结构定义 ---------------------------


class RootCauseInfo(BaseModel):
    """根因信息：描述单条异常的可能根因"""

    metric: str = Field(
        default="",
        description="异常指标名称（例如：gala_gopher_sli_container_cpu_busy）",
    )
    labels: Dict[str, Union[str, int, float]] = Field(
        default_factory=dict,
        description="根因指标的详细标签，如container_name、machine_id等",
    )
    score: float = Field(default=0.0, description="异常得分（相关性或异常度分数）")


class RootCauseResult(BaseModel):
    """根因分析结果"""

    rootcause_info: List[RootCauseInfo] = Field(
        default_factory=list, description="异常根因信息列表（默认空列表）"
    )
    start_time: int = Field(default=0, description="检测窗口开始时间戳（秒）")
    end_time: int = Field(default=0, description="检测窗口结束时间戳（秒）")


class AnomalyInfo(BaseModel):
    """检测阶段的异常详情"""

    machine_id: str = Field(default="", description="机器ID（如：UUID或IP）")
    metric: str = Field(default="", description="异常指标名称")
    labels: Dict[str, Union[str, int, float]] = Field(
        default_factory=dict,
        description="异常指标的标签信息（container_id、instance等）",
    )
    score: float = Field(default=0.0, description="异常分数（越高说明异常程度越高）")
    entity_name: Optional[str] = Field(
        default="", description="实体名（如指标的entity_name）"
    )
    details: Dict[str, Union[str, int, float, dict]] = Field(
        default_factory=dict, description="检测附加信息（例如容器额外信息）"
    )


class PerceptionResult(BaseModel):
    """容器干扰检测结果"""

    is_anomaly: bool = Field(default=False, description="是否存在异常")
    anomaly_info: List[AnomalyInfo] = Field(
        default_factory=list, description="异常节点详细列表"
    )
    start_time: int = Field(default=0, description="检测窗口开始时间戳（秒）")
    end_time: int = Field(default=0, description="检测窗口结束时间戳（秒）")


# --------------------------- 运行参数定义 ---------------------------


class KPIParam(BaseModel):
    """KPI配置参数"""

    metric: str
    entity_name: Optional[str] = ""
    params: Dict[str, Any] = Field(default_factory=dict)


class WindowParam(BaseModel):
    """检测窗口参数"""

    look_back: int = 20  # 向前回溯分钟数
    obs_size: int = 6  # 观测点数量


class ExtraConfig(BaseModel):
    """额外检测配置"""

    extra_metrics: str = ""


# --------------------------- 报告与接口定义 ---------------------------


class ReportType(str, Enum):
    """报告类型"""

    normal = "normal"
    anomaly = "anomaly"
