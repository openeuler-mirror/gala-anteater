from __future__ import annotations
import os
import sys
import json
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime

from common.loader import build_metric_loader

from mcp.server import FastMCP
from anteater.core.anomaly import Anomaly
from anteater.core.ts import TimeSeries
from anteater.core.kpi import KPI, ModelConfig
from anteater.model.detector.disruption_detector import ContainerDisruptionDetector
from anteater.utils.common import GlobalVariable
from container_disruption_detection.container_disruption_detection_mcp.report_api import generate_degraded_report, generate_normal_report

from container_disruption_detection.container_disruption_detection_mcp.mcp_data import (
    RootCauseResult,
    RootCauseInfo,
    AnomalyInfo,
    PerceptionResult,
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


def render_report(
    anomalies: PerceptionResult, report_type: ReportType
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
            3. 调用完成后如果出现容器干扰现象，则把当前工具得到的结果作为入参，调用root_cause_analysis_tool方法 ，如果没有出现劣化现象，则调用报告工具返回报告给用户。\
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
) -> PerceptionResult:
    """容器异常检测工具（支持自动识别机器ID）"""
    job_path = os.path.join(os.path.dirname(__file__), "../config/container_disruption.job.json")
    anteater_conf = os.path.join(os.path.dirname(__file__), "../config/gala-anteater.yaml")
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
        anomaly_result.start_time = start
        anomaly_result.end_time = end
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
                        metric=cause.metric,
                        labels=cause.labels,
                        score=cause.score
                    )
                )
        
        anomaly_infos.append(
            AnomalyInfo(
                machine_id=anomaly.machine_id,
                metric=anomaly.metric,
                labels=anomaly.labels,
                score=anomaly.score,
                entity_name=anomaly.entity_name,
                details=anomaly.details or {}
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
    3. 本方法得到的结果必须再调用generate_report 生成报告给到用户")
@mcp.tool(name="root_cause_analysis_tool")
def root_cause_analysis_tool(
    anomalies: PerceptionResult,
    start_time: str,
    end_time: str,
    window: WindowParam = WindowParam(),
    anteater_conf: Optional[str] = None,
    metric_info: Optional[dict] = None,
    machine_id: str = "",
) -> RootCauseResult:
    if anomalies.is_anomaly == False:
        raise ValueError("当container_disruption_detection_tool检测出异常事件时，再调用此工具")
    job_path = os.path.join(os.path.dirname(__file__), "../config/container_disruption.job.json")
    anteater_conf = os.path.join(os.path.dirname(__file__), "../config/gala-anteater.yaml")
    kpis, window, extra = load_kpis_from_job(job_path)

    loader = build_metric_loader(config_path = anteater_conf, metricinfo_json=metric_info)
    facade = ContainerDisruptionFacade(loader, ExtraConfig())

    metric = anomalies.anomaly_info[0].metric
    machine_id = anomalies.anomaly_info[0].machine_id
    # 最好打log看看，start_time和end_time是不是和container_disruption_detection_tool的一致
    if GlobalVariable.is_test_model == False:
        GlobalVariable.is_test_model = True
        GlobalVariable.start_time = start_time
        GlobalVariable.end_time = end_time
        _, ts_list = facade.get_kpi_ts_list(metric, machine_id, window.look_back)
        GlobalVariable.is_test_model = False
    else:
        _, ts_list = facade.get_kpi_ts_list(metric, machine_id, window.look_back)
    
    rootcauses = []
    # todo，这里全部都判断了一遍，有些冗余，需要从container_disruption_detection_tool返回看哪些_ts存在异常
    for _ts in ts_list:
        rootcauses.extend(facade.find_disruption_source(_ts, ts_list))

    rootcause_infos = []
    for rootcause in rootcauses:
        rootcause_infos.append(
            RootCauseInfo(
                metric=rootcause.metric,
                labels=rootcause.labels,
                score=rootcause.score
            )
        )
    
    rootcause_result = RootCauseResult(
        start_time=start_time,
        end_time=end_time,
        rootcause_info=rootcause_infos
    )
    
    return rootcause_result

@mcp.prompt(description="调用container_disruption_detection_tool 或 root_cause_analysis_tool 后把结果传入generate_report ")
@mcp.tool(name="report_tool")
def generate_report_tool(
    source_data: Union[PerceptionResult, RootCauseResult], report_type: ReportType
) -> dict:
    """
    使用 报告工具：生成最终Markdown格式报告
    输入:
    source_data 感知或定界的结果
    report_type 是否劣化 normal anomaly
    您是一个专业的容器干扰检测分析人员，擅长分析容器之间性能干扰情况，生成报告，报告标题“容器干扰检测诊断报告”。以下内容如实回答，不要发散。
    先判断是否存在容器性能干扰，{report_type}为normal 不存在容器干扰，anomaly 存在容器干扰；
    未劣化分析步骤如下：
    1、总览：根据<context>里的{start_time}{end_time}得到开始和结束时间，结论是当前AI训练任务运行正常，将持续监测。
    劣化分析步骤如下：
    1、总览：根据<context>里的{time}得到检测时间，{abnormalNodeCount}异常节点数量，{compute}{network}{storage}异常类型true为异常，false正常；
    2、细节：每条节点的具体卡号{objectId}、异常指标{kpiId}（其中：HcclAllGather表示集合通信库的AllGather时序序列指标；HcclReduceScatter表示集合通信库的ReduceScatter时序序列指标；HcclAllReduce表示集合通信库的AllReduce时序序列指标；），检测方法{methodType}（SPACE 多节点空间对比检测器，TIME 单节点时间检测器），以表格形式呈现；
    3、针对这个节点给出检测建议，如果是计算类型，建议检测卡的状态，算子下发以及算子执行的代码，对慢节点进行隔离；如果是网络问题，建议检测组网的状态，使用压测节点之间的连通状态；如果是存储问题，建议检测存储的磁盘以及用户脚本中的dataloader和保存模型代码。
    """
    logger.info("调用 报告工具，report_type = " + report_type.value)
    # 根据报告类型调用对应的生成方法
    if report_type == ReportType.normal:
        result = generate_normal_report(source_data)
    elif report_type == ReportType.anomaly:
        result = generate_degraded_report(source_data)
    else:
        raise Exception("不支持的报告类型")
    logger.info(f"报告：{result}")
    return result
    # render_report


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