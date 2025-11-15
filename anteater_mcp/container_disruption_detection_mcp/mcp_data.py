from __future__ import annotations
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from enum import Enum

# ============================================================
#                     公共枚举 & 基础结构
# ============================================================

class ReportType(str, Enum):
    normal = "normal"
    anomaly = "anomaly"


# ============================================================
#                Root Cause（干扰源分析相关）
# ============================================================

class RootCauseInfo(BaseModel):
    """干扰源分析的根因项"""
    metric: str = Field(default="", description="根因指标 ID")
    labels: Dict[str, Union[str, int, float]] = Field(
        default_factory=dict, description="指标标签"
    )
    score: float = Field(default=0.0, description="相关性得分（0~1）")


class RootCauseResult(BaseModel):
    """一次根因分析的完整结果"""
    rootcause_info: List[RootCauseInfo] = Field(
        default_factory=list, description="根因列表"
    )
    start_time: int = Field(default=0, description="分析起始时间（秒）")
    end_time: int = Field(default=0, description="分析结束时间（秒）")


# ============================================================
#               Detection 阶段（异常感知，内部结构）
# ============================================================

class AnomalyInfo(BaseModel):
    """检测阶段发现的 单条 异常项（内部格式）"""
    machine_id: str = Field(default="", description="机器 ID")
    metric: str = Field(default="", description="SLI 指标名称")
    labels: Dict[str, Any] = Field(default_factory=dict, description="指标标签")
    score: float = Field(default=0.0, description="异常得分 0~1")
    entity_name: str = Field(default="", description="实体名（容器名等）")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="检测阶段生成的额外信息"
    )


class PerceptionResult(BaseModel):
    """内部：容器干扰检测结果（对应 server 的 perception）"""
    is_anomaly: bool = Field(default=False, description="是否检测到异常")
    anomaly_info: List[AnomalyInfo] = Field(
        default_factory=list, description="异常列表"
    )
    start_time: int = Field(default=0, description="检测窗口开始时间（秒）")
    end_time: int = Field(default=0, description="检测窗口结束时间（秒）")


# ============================================================
#                   KPI & 窗口定义（运行参数）
# ============================================================

class KPIParam(BaseModel):
    metric: str = Field(default="", description="指标名称")
    entity_name: str = Field(default="", description="实体名（容器）")
    params: Dict[str, Any] = Field(default_factory=dict, description="检测参数")


class WindowParam(BaseModel):
    look_back: int = Field(default=20, description="回溯时间（分钟）")
    obs_size: int = Field(default=6, description="测试窗口大小")


class ExtraConfig(BaseModel):
    """额外配置（例如 extra_metrics）"""
    extra_metrics: str = Field(default="", description="需要额外采集的指标")


# ============================================================
#       MCP 对外报告结构（检测 / 分析 / 恢复建议）
# ============================================================

# -------------------------
# 检测报告（对应 detection_report）
# -------------------------

class DetectionDetailItem(BaseModel):
    container_id: str = Field(default="", description="存在干扰的容器 ID")
    metric_id: str = Field(default="", description="存在干扰的指标 ID")
    start_timestamp: int = Field(default=0, description="异常开始时间（毫秒）")
    end_timestamp: int = Field(default=0, description="异常结束时间（毫秒）")
    abnormal_level: str = Field(
        default="轻度", description="异常级别（轻度/中度/严重）"
    )
    disruption_peak: int = Field(default=0, description="异常峰值")


class DetectionReport(BaseModel):
    metric_list: List[str] = Field(
        default_factory=list, description="当前任务分析过的 metric 列表"
    )
    disruption_cnt: int = Field(
        default=0, description="检测出被干扰症状的 metric 数量"
    )
    details: List[DetectionDetailItem] = Field(
        default_factory=list, description="检测详情列表"
    )


# -------------------------
# 干扰源分析报告（对应 analysis_report）
# -------------------------

class AnalysisDetailItem(BaseModel):
    container_id: str = Field(default="", description="受害容器 ID")
    disrupted_metric_id: str = Field(default="", description="被干扰的指标")
    # 文档中“类型”列写了 list，但说明是“字典键为干扰源容器ID，值为概率值”，
    # 这里按说明实现为 Dict[str, float]
    interf_src_probs: Dict[str, float] = Field(
        default_factory=dict,
        description="干扰源概率分布，键为容器 ID，值为 [0,1] 的概率",
    )


class AnalysisReport(BaseModel):
    metric_list: List[str] = Field(
        default_factory=list, description="分析过的 metric 列表"
    )
    disruption_cnt: int = Field(
        default=0, description="检测到的干扰源数量"
    )
    details: List[AnalysisDetailItem] = Field(
        default_factory=list, description="干扰源分析详情"
    )


# -------------------------
# 恢复建议报告（对应 recovery_suggestion）
# -------------------------

class RecoverySuggestionItem(BaseModel):
    suggestion: str = Field(default="", description="恢复建议内容")
    evidence: str = Field(default="", description="建议依据的证据说明")
    example: str = Field(default="", description="建议的具体操作示例")


class RecoveryReport(BaseModel):
    # 按 API 文档命名：recovery_suggestion 为列表
    recovery_suggestion: List[RecoverySuggestionItem] = Field(
        default_factory=list, description="恢复建议列表"
    )
