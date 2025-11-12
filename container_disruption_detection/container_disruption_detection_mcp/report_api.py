import json
from datetime import datetime
from container_disruption_detection.container_disruption_detection_mcp.mcp_data import (
    PerceptionResult,
    RootCauseResult
)


def _format_timestamp(ts: int) -> str:
    """内部工具函数：时间戳转格式化字符串"""
    if not ts:
        return "未知时间"
    try:
        # 时间戳是秒，不需要除以1000
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def generate_normal_report(data: PerceptionResult) -> dict:
    """生成正常报告"""
    start = _format_timestamp(data.start_time)
    end = _format_timestamp(data.end_time)
    return {
        "reportName": "容器干扰检测诊断报告",
        "overview": {
            "start_time": start,
            "end_time": end,
            "status": "正常",
            "description": "当前AI任务运行正常，将持续监测。"
        },
        "details": []
    }


def generate_degraded_report(data: RootCauseResult) -> dict:
    """生成异常报告"""
    start = _format_timestamp(data.start_time)
    end = _format_timestamp(data.end_time)
    rootcauseInfos = data.rootcause_info or []

    details = [
        {"metric": rca.metric, "labels": rca.labels, "score": rca.score}
        for rca in rootcauseInfos
    ]

    return {
        "reportName": "容器干扰检测诊断报告",
        "overview": {
            "start_time": start,
            "end_time": end,
            "status": "存在异常",
            "abnormalNodeCount": len(details),
            "description": "检测到容器存在性能干扰，请参考下方详细信息。"
        },
        "details": details
    }
