from __future__ import annotations
import os
import sys
import logging
from typing import List, Optional, Union

from common.loader import build_metric_loader

from mcp.server import FastMCP
from anteater.core.anomaly import Anomaly
from anteater.core.ts import TimeSeries
from anteater.core.kpi import KPI, ModelConfig
from anteater.model.detector.disruption_detector import ContainerDisruptionDetector
from anteater.utils.common import GlobalVariable
from anteater_mcp.container_disruption_detection_mcp.report_api import (
    generate_degraded_report,
    generate_normal_report,
)

from anteater_mcp.container_disruption_detection_mcp.mcp_data import (
    RootCauseResult,
    RootCauseInfo,
    AnomalyInfo,
    PerceptionResult,
    KPIParam,
    WindowParam,
    ExtraConfig,
    ReportType,
)

from anteater_mcp.container_disruption_detection_mcp.utils import (
    load_kpis_from_job,
    dt_last,
)

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
        return start, end, self.detector.get_unique_machine_id(start, end, kpis)

    def get_kpi_ts_list(self, metric: str, machine_id: str, look_back: int):
        return self.detector.get_kpi_ts_list(metric, machine_id, look_back)

    def detect_by_spot(self, kpi, machine_id: str) -> List[Anomaly]:
        return self.detector.detect_by_spot(kpi, machine_id)

    def find_disruption_source(self, victim_ts: TimeSeries, all_ts: List[TimeSeries]):
        return self.detector.find_disruption_source(victim_ts, all_ts)

    def get_container_extra_info(
        self, machine_id, container_name, start_time, end_time, obs_size
    ):
        return self.detector.get_container_extra_info(
            machine_id, container_name, start_time, end_time, obs_size
        )


@mcp.prompt(
    description="调用逻辑:1. 当用户询问特定容器ID的容器性能是否被干扰时调用。2. 检测结果将决定后续流程走向。\
            3. 调用完成后如果出现容器干扰现象，则把当前工具得到的结果作为入参，调用root_cause_analysis_tool方法 ，如果没有出现劣化现象，则调用报告工具返回报告给用户。\
            4. 本方法得到的结果必须再调用 report_tool 生成报告给到用户"
)
@mcp.tool(name="container_disruption_detection_tool")
def container_disruption_detection_tool(
    kpis: List[KPIParam] = None,
    window: WindowParam = WindowParam(),
    extra: Optional[ExtraConfig] = None,
    anteater_conf: Optional[str] = None,
    metric_info: Optional[dict] = None,
    machine_id: Optional[str] = None,
) -> PerceptionResult:
    """容器异常检测工具（支持自动识别机器ID）"""
    job_path = os.path.join(
        os.path.dirname(__file__), "../config/container_disruption.job.json"
    )
    anteater_conf = os.path.join(
        os.path.dirname(__file__), "../config/gala-anteater.yaml"
    )
    kpis, window, extra = load_kpis_from_job(job_path)

    loader = build_metric_loader(config_path=anteater_conf, metricinfo_json=metric_info)
    facade = ContainerDisruptionFacade(
        loader,
        ModelConfig(
            name="container_disruption_detection",
            model_path="../../anteater/model/detector/disruption_detector.py",
            params={"extra_metrics": extra.extra_metrics},
        ),
    )
    anomalies = []
    anomaly_result = PerceptionResult(
        is_anomaly=False,
        anomaly_info=[],
    )

    if not machine_id:
        if not kpis:
            raise ValueError("必须提供至少一个 KPI 参数")
        start, end, machine_ids = facade.get_unique_machine_id(window.look_back, kpis)
        anomaly_result.start_time = int(start.timestamp())
        anomaly_result.end_time = int(end.timestamp())
    else:
        machine_ids = [machine_id]

    for mid in machine_ids:
        for k in kpis:
            anomalies.extend(facade.detect_by_spot(k, mid))

    logger.info("检测完成，总机器数: %d", len(machine_ids))

    # 将Anomaly对象转换为AnomalyInfo对象
    anomaly_infos = []
    for anomaly in anomalies:
        # 转换root_causes为RootCauseResult对象
        root_causes = []
        if anomaly.root_causes:
            for cause in anomaly.root_causes:
                root_causes.append(
                    RootCauseResult(
                        metric=cause.metric, labels=cause.labels, score=cause.score
                    )
                )

        anomaly_infos.append(
            AnomalyInfo(
                machine_id=anomaly.machine_id,
                metric=anomaly.metric,
                labels=anomaly.labels,
                score=anomaly.score,
                entity_name=anomaly.entity_name,
                details=anomaly.details or {},
            )
        )

    # 构造并返回PerceptionResult对象
    if anomaly_infos:
        anomaly_result.is_anomaly = True
        anomaly_result.anomaly_info = anomaly_infos

    return anomaly_result


@mcp.prompt(
    description="调用逻辑:1. 仅在容器干扰检测工具返回is_anomaly=True时调用。 \
    2. 接收感知工具的全量性能数据作为输入。  \
    3. 本方法得到的结果必须再调用 report_tool 生成报告给到用户"
)
@mcp.tool(name="root_cause_analysis_tool")
def root_cause_analysis_tool(
    anomalies: PerceptionResult,
    window: WindowParam = WindowParam(),
    anteater_conf: Optional[str] = None,
    metric_info: Optional[dict] = None,
    machine_id: str = "",
) -> RootCauseResult:
    if anomalies.is_anomaly == False:
        raise ValueError(
            "当container_disruption_detection_tool检测出异常事件时，再调用此工具"
        )
    job_path = os.path.join(
        os.path.dirname(__file__), "../config/container_disruption.job.json"
    )
    anteater_conf = os.path.join(
        os.path.dirname(__file__), "../config/gala-anteater.yaml"
    )
    kpis, window, extra = load_kpis_from_job(job_path)

    loader = build_metric_loader(config_path=anteater_conf, metricinfo_json=metric_info)
    facade = ContainerDisruptionFacade(
        loader,
        ModelConfig(
            name="container_disruption_detection",
            model_path="../../anteater/model/detector/disruption_detector.py",
            params={"extra_metrics": extra.extra_metrics},
        ),
    )

    # 使用第一个异常作为入口点获取同一机器的时序列表
    metric = anomalies.anomaly_info[0].metric
    machine_id = anomalies.anomaly_info[0].machine_id
    if GlobalVariable.is_test_model == False:
        GlobalVariable.is_test_model = True
        from datetime import datetime

        GlobalVariable.start_time = datetime.fromtimestamp(anomalies.start_time)
        GlobalVariable.end_time = datetime.fromtimestamp(anomalies.end_time)
        _, ts_list = facade.get_kpi_ts_list(metric, machine_id, window.look_back)
        GlobalVariable.is_test_model = False
    else:
        _, ts_list = facade.get_kpi_ts_list(metric, machine_id, window.look_back)

    # 只分析检测阶段判定为异常的时间序列
    abnormal_keys = {(a.metric, a.machine_id) for a in anomalies.anomaly_info}

    rootcauses = []

    for _ts in ts_list:
        ts_metric = getattr(_ts, "metric", None)
        ts_machine_id = _ts.labels.get("machine_id") if hasattr(_ts, "labels") else None

        if (ts_metric, ts_machine_id) in abnormal_keys:
            logger.info(
                f"根因分析: 检测到异常时序 metric={ts_metric}, machine_id={ts_machine_id}"
            )
            rootcauses.extend(facade.find_disruption_source(_ts, ts_list))
        else:
            logger.debug(f"跳过正常时序 metric={ts_metric}, machine_id={ts_machine_id}")

    rootcause_infos = [
        RootCauseInfo(metric=rc.metric, labels=rc.labels, score=rc.score)
        for rc in rootcauses
    ]

    rootcause_result = RootCauseResult(
        rootcause_info=rootcause_infos,
        start_time=anomalies.start_time,
        end_time=anomalies.end_time,
    )

    return rootcause_result


@mcp.prompt(
    description="""
调用逻辑：
1. 仅在 container_disruption_detection_tool 或 root_cause_analysis_tool 执行完成后调用；
2. 当检测结果正常（无容器性能干扰）时，传入 report_type=normal；
   当检测结果存在性能干扰时，传入 report_type=anomaly；
3. 本工具生成结构化报告数据，供大模型后续自然语言报告生成使用；
4. 不论是否存在干扰，执行完检测或根因分析后都必须调用此工具。
"""
)
@mcp.tool(name="report_tool")
def generate_report_tool(
    source_data: Union[PerceptionResult, RootCauseResult], report_type: ReportType
) -> dict:
    """
    容器干扰诊断报告生成工具

    角色设定：
    您是一名专业的容器性能干扰诊断工程师，
    擅长分析容器化环境和 AI 训练任务的性能干扰问题。
    您将基于 <context> 中的检测或根因分析结果生成
    “容器干扰检测诊断报告”，内容必须真实、专业、可供研发与运维使用。

    写作要求：
    - 报告标题固定为“容器干扰检测诊断报告”；
    - 若 report_type=normal，则直接说明系统运行正常；
    - 若 report_type=anomaly，则说明干扰类型、异常容器数量及其指标；
    - 禁止出现 GPU / CPU / 磁盘 等无关字眼；
    - 当前时间可参考 {{ time }}；
    - 报告由大模型最终生成自然语言文本，本函数仅返回结构化 JSON 数据。

    报告逻辑结构：
    1. **总览**：检测时间范围、状态（正常 / 异常）、异常数量；
    2. **细节**：异常容器的指标 (metric)、标签 (labels)、得分 (score)；
    3. **建议**：
        - 若为计算类异常：检查算子执行逻辑与任务下发；
        - 若为网络类异常：检查容器组网与端口延迟；
        - 若为存储类异常：检查I/O性能与数据加载逻辑。

    输出：
    - 返回 JSON 格式的结构化报告；
    - 大模型基于本结构生成最终人类可读报告。
    """
    logger.info(f"调用 报告工具，report_type = {report_type.value}")

    # 根据报告类型生成对应结构
    if report_type == ReportType.normal:
        result = generate_normal_report(source_data)
    elif report_type == ReportType.anomaly:
        result = generate_degraded_report(source_data)
    else:
        raise ValueError(f"不支持的报告类型: {report_type}")

    logger.info(f"报告生成完成: {result.get('overview', {}).get('status')}")
    return result


if __name__ == "__main__":
    if os.name == "posix":
        import multiprocessing

        multiprocessing.set_start_method("spawn", force=True)
    """
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
    """

    mcp.run(transport="sse")
