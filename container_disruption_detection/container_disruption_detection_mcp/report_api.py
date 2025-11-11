import json
from datetime import datetime

from container_disruption_detection.container_disruption_detection_mcp.mcp_data import (
    PerceptionResult,
    RootCauseResult
)


def generate_normal_report(data: PerceptionResult) -> dict:
    """生成无劣化的正常报告"""
    data = data.model_dump()
    timestamp = data.get("start_time")
    start_time = datetime.fromtimestamp(timestamp // 1000).strftime("%Y-%m-%d %H:%M:%S") if timestamp else "未知时间"
    timestamp = data.get("end_time")
    end_time = datetime.fromtimestamp(timestamp // 1000).strftime("%Y-%m-%d %H:%M:%S") if timestamp else "未知时间"
    data["start_time"] = start_time
    data["end_time"] = end_time

    return data

# todo, 待完善
def generate_degraded_report(data: RootCauseResult) -> dict:
    """
        生成设备异常状态的JSON报告

        参数:
            data: 包含设备状态信息的字典

        返回:
            格式化的JSON报告字典
        """
    # 解析时间戳为可读格式
    data = data.model_dump()
    start_time = data.get("strat_time")
    end_time = data.get("end_time")
    
    # 提取异常信息
    rootcauseInfos = data.get("rootcause_info", [])

    # 整理异常节点详情
    abnormal_nodes = []
    for rootcauseInfo in rootcauseInfos:
        abnormal_nodes.append({
            "metric": rootcauseInfo.get("metric"),
            "labels": rootcauseInfo.get("labels"),
            "score": rootcauseInfo.get("score")
        })

    # 构建JSON报告
    report = {
        "reportName": "容器干扰检测诊断报告",
        "overview": {
            "start_time": start_time,
            "end_time": end_time,
            "abnormal_nodes": abnormal_nodes
        }
    }

    return report

