from __future__ import annotations
import os
import sys
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

from common.loader import build_metric_loader

from mcp.server import FastMCP
from anteater.core.anomaly import Anomaly
from anteater.core.ts import TimeSeries
from anteater.core.kpi import KPI, ModelConfig
from anteater.model.detector.disruption_detector import ContainerDisruptionDetector

from container_disruption_detection.container_disruption_detection_mcp.mcp_data import (
    RootCauseModel,
    AnomalyModel,
    KPIParam,
    WindowParam,
    ExtraConfig,
    ReportType,
)

from container_disruption_detection.container_disruption_detection_mcp.utils import load_kpis_from_job, dt_last

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("container_disruption_detection_mcp")

mcp = FastMCP("Container Disruption Detection MCP", host="0.0.0.0", port=12345)


class ContainerDisruptionFacade:
    def __init__(self, data_loader, config: ExtraConfig):
        self.data_loader = data_loader
        self.config = config
        self.detector = ContainerDisruptionDetector(data_loader, config)

    def get_unique_machine_id(self, look_back: int, kpis: List[KPI]) -> List[str]:
        start, end = dt_last(minutes=look_back)
        return self.detector.get_unique_machine_id(start, end, kpis)

    def get_kpi_ts_list(self, metric: str, machine_id: str, look_back: int):
        return self.detector.get_kpi_ts_list(metric, machine_id, look_back)

    def detect_by_spot(self, kpi, machine_id: str) -> List[Anomaly]:
        return self.detector.detect_by_spot(kpi, machine_id)

    def find_disruption_source(self, victim_ts: TimeSeries, all_ts: List[TimeSeries]):
        return self.detector.find_discruption_source(victim_ts, all_ts)

    def get_container_extra_info(
        self, machine_id, container_name, start_time, end_time, obs_size
    ):
        return self.detector.get_container_extra_info(
            machine_id, container_name, start_time, end_time, obs_size
        )


def render_report(
    anomalies: List[AnomalyModel], report_type: ReportType
) -> Dict[str, str]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if report_type == ReportType.normal or not anomalies:
        md = [
            "# 容器干扰检测诊断报告",
            f"**时间**：{now}",
            "## 总览",
            "当前容器运行正常，将持续监测。",
        ]
        return {"markdown": "\n\n".join(md)}

    md = [
        "# 容器干扰检测诊断报告",
        f"**时间**：{now}",
        "## 总览",
        f"检测到异常容器数量：**{len(anomalies)}**",
        "## 细节",
        "| 机器 | 指标 | 分数 | 容器 | 细节 | RCA |",
        "|---|---:|---:|---|---|---|",
    ]
    for a in anomalies:
        container = a.labels.get("container_name", "")
        rca_txt = (
            ", ".join([f"{rc.metric}({rc.score})" for rc in a.root_causes])
            if a.root_causes
            else "-"
        )
        md.append(
            f"| {a.machine_id} | {a.metric} | {a.score:.3f} | {container} | {json.dumps(a.details.get('info', {}), ensure_ascii=False)} | {rca_txt} |"
        )

    md.append("## 建议\n- 请检查计算、网络、存储链路，隔离慢节点。")
    return {"markdown": "\n\n".join(md)}


@mcp.prompt(description="调用逻辑:1. 当用户询问特定容器ID的容器性能是否被干扰时调用。2. 检测结果将决定后续流程走向。\
            3. 调用完成后如果出现容器干扰现象，则把当前工具得到的结果作为入参，调用rca_tool方法 ，如果没有出现劣化现象，则调用报告工具返回报告给用户。\
            4. 本方法得到的结果必须再调用generate_report 生成报告给到用户"
            )
@mcp.tool(name="container_disruption_detection_tool")
def container_disruption_detection_tool(
    kpis: List[KPIParam] = None,
    window: WindowParam = WindowParam(),
    extra: Optional[ExtraConfig] = None,
    anteater_conf: Optional[str] = None,
    metric_info: Optional[dict] = None,
    machine_id: Optional[str] = None,
) -> List[AnomalyModel]:
    """容器异常检测工具（支持自动识别机器ID）"""
    job_path = os.path.join(os.path.dirname(__file__), "../config/container_disruption.job.json")
    anteater_conf = os.path.join(os.path.dirname(__file__), "../config/gala-anteater.yaml")
    kpis, window, extra = load_kpis_from_job(job_path)
    print(f"kpis: {kpis}, window: {window}, extra: {extra}")

    loader = build_metric_loader(config_path=anteater_conf, metricinfo_json=metric_info)
    facade = ContainerDisruptionFacade(
        loader,
        ModelConfig(
            name="container_disruption_detection",
            model_path="../../anteater/model/detector/disruption_detector.py",
            params={"extra_metrics": extra.extra_metrics},
        ),
    )
    anomalies: List[AnomalyModel] = []

    if not machine_id:
        if not kpis:
            raise ValueError("必须提供至少一个 KPI 参数")
        machine_ids = facade.get_unique_machine_id(window.look_back, kpis)
    else:
        machine_ids = [machine_id]

    for mid in machine_ids:
        for k in kpis:
            anomalies.extend(facade.detect_by_spot(k, mid))

    logger.info("检测完成，总机器数: %d", len(machine_ids))

    # 将AnomalyModel类型转为字典
    result = []
    for anomaly in anomalies:
        result.append({
            'machine_id': anomaly.machine_id,
            'metric': anomaly.metric,
            'labels': anomaly.labels,
            'score': float(anomaly.score),  # 将 np.float64 转换为 Python float
            'entity_name': anomaly.entity_name,
            'details': anomaly.details,
        })
    
    return result
    # return anomalies

@mcp.prompt(
    description="调用逻辑:1. 仅在容器干扰检测工具返回is_anomaly=True时调用。 \
    2. 接收感知工具的全量性能数据作为输入。  \
    3. 本方法得到的结果必须再调用generate_report 生成报告给到用户")
@mcp.tool(name="rca_tool")
def rca_tool(
    metric: str,
    victim_container_name: str,
    window: WindowParam = WindowParam(),
    anteater_conf: Optional[str] = None,
    metric_info: Optional[dict] = None,
    machine_id: str = "",
) -> List[RootCauseModel]:
    if not machine_id:
        raise ValueError("rca_tool 需要提供 machine_id")

    loader = build_metric_loader(config_path=anteater_conf, metricinfo_json=metric_info)
    facade = ContainerDisruptionFacade(loader, ExtraConfig())
    _, ts_list = facade.get_kpi_ts_list(metric, machine_id, window.look_back)
    victim_list = [
        ts for ts in ts_list if ts.labels.get("container_name") == victim_container_name
    ]
    if not victim_list:
        raise RuntimeError(f"未找到容器 {victim_container_name} 的时序")

    return facade.find_disruption_source(victim_list[0], ts_list)


@mcp.tool(name="report_tool")
def report_tool(
    anomalies: List[AnomalyModel], report_type: ReportType = ReportType.anomaly
):
    return render_report(anomalies, report_type)


if __name__ == "__main__":
    if os.name == "posix":
        import multiprocessing

        multiprocessing.set_start_method("spawn", force=True)

    '''
    job_path = os.path.join(
        os.path.dirname(__file__), "../config/container_disruption.job.json"
    )
    anteater_conf_path = os.path.join(
        os.path.dirname(__file__), "../config/gala-anteater.yaml"
    )

    kpis, window, extra = load_kpis_from_job(job_path)
    logger.info("配置加载成功，开始检测。")

    metric_info = {}

    anomalies = container_disruption_detection_tool(
        kpis=kpis,
        window=window,
        extra=extra,
        anteater_conf=anteater_conf_path,
        metric_info=metric_info,
    )
    '''
    
    mcp.run(transport="sse")
