from __future__ import annotations
import os
import sys
import json
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
import numpy as np
import pandas as pd

from common.loader import build_metric_loader

from mcp.server import FastMCP
from anteater.core.ts import TimeSeries
from anteater.model.algorithms.normalization import Normalization
from anteater.model.algorithms.spot import Spot
from anteater.model.algorithms.ts_dbscan import TSDBSCAN

from mcp_data import (
    RootCauseModel,
    AnomalyModel,
    KPIParam,
    WindowParam,
    ExtraConfig,
    ReportType,
)

from utils import load_kpis_from_job, load_anteater_conf, divide, dt_last

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("container_disruption_detection_mcp")

mcp = FastMCP("Container Disruption Detection MCP", host="0.0.0.0", port=12345)


# Facade 主体类
class ContainerDisruptionFacade:
    def __init__(self, data_loader, config: ExtraConfig):
        self.data_loader = data_loader
        self.config = config
        self.q = 1e-3
        self.level = 0.98
        self.smooth_win = 3
        self.container_num = 0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    # 自动获取机器ID
    def get_unique_machine_ids(self, look_back: int, kpis: List[KPIParam]) -> List[str]:
        """自动获取在指定时间窗口内活跃的机器ID"""
        start, end = dt_last(minutes=look_back)
        metrics = [k.metric for k in kpis]

        try:
            machine_ids = self.data_loader.get_unique_machines(start, end, metrics)
        except Exception as e:
            logger.warning(
                f"get_unique_machines() 调用失败，将使用 fallback 方式提取 machine_id: {e}"
            )
            ts_list = self.data_loader.get_metric(start, end, metrics[0])
            machine_ids = list(
                {
                    ts.labels.get("machine_id", "")
                    for ts in ts_list
                    if ts.labels.get("machine_id")
                }
            )

        if not machine_ids:
            logger.warning("未检测到任何 machine_id，请检查数据源配置或时间窗口。")
        else:
            logger.info(f"自动检测到 {len(machine_ids)} 台机器: {machine_ids}")
        return machine_ids

    # 在线取数
    def get_kpi_ts_list(self, metric: str, machine_id: str, look_back: int):
        start, end = dt_last(minutes=look_back)
        self.start_time, self.end_time = start, end
        point_count = self.data_loader.expected_point_length(start, end)
        ts_list = self.data_loader.get_metric(start, end, metric, machine_id=machine_id)
        return point_count, ts_list

    def detect_by_spot(
        self,
        metric: str,
        machine_id: str,
        outlier_ratio_th: float,
        look_back: int,
        obs_size: int,
    ) -> List[AnomalyModel]:
        ts_dbscan_detector = TSDBSCAN(look_back=look_back, obs_size=obs_size)
        point_count, ts_list = self.get_kpi_ts_list(metric, machine_id, look_back)

        anomalies: List[AnomalyModel] = []
        self.container_num += len(ts_list)

        for _ts in ts_list:
            detect_result = ts_dbscan_detector.detect(_ts.values)
            train_data = [
                _ts.values[i]
                for i in range(len(detect_result))
                if detect_result[i] == 0
            ]
            test_data = _ts.values[-obs_size:]

            if not self._is_data_valid(_ts.values, point_count, obs_size):
                score = 0.0
            else:
                result = self._spot_detect(train_data, test_data, obs_size)
                score = float(divide(result, obs_size))

            if score >= outlier_ratio_th:
                extra_info = (
                    self.get_container_extra_info(
                        machine_id,
                        _ts.labels.get("container_name", ""),
                        self.start_time,
                        self.end_time,
                        obs_size,
                    )
                    if (self.start_time and self.end_time)
                    else {}
                )
                anomalies.append(
                    AnomalyModel(
                        machine_id=machine_id,
                        metric=_ts.metric,
                        labels=_ts.labels,
                        score=score,
                        entity_name="container",
                        details={"event_source": "spot", "info": extra_info},
                    )
                )
        return anomalies

    def _spot_detect(self, train_data, test_data, obs_size):
        spot = Spot(q=self.q)
        level = self._check_level(np.array(train_data), self.level)
        spot.initialize(train_data, level=level)
        test_data, _, _ = Normalization.clip_transform(
            np.array(test_data)[np.newaxis, :], is_clip=False
        )
        thr_with_alarms = spot.run(test_data[0], with_alarm=True)
        bound_result = np.array(
            test_data[0] > thr_with_alarms["thresholds"], dtype=np.int32
        )
        return int(np.sum(bound_result))

    def find_disruption_source(
        self, victim_ts: TimeSeries, all_ts: List[TimeSeries]
    ) -> List[RootCauseModel]:
        tmp_causes: List[RootCauseModel] = []
        for ts in all_ts:
            if ts is victim_ts:
                continue
            df = pd.DataFrame({"victim": victim_ts.values, "source": ts.values})
            self._normalize_df(df)
            corr = abs(df.corr().iloc[0, 1])
            if corr > 0.5:
                tmp_causes.append(
                    RootCauseModel(
                        metric=ts.metric, labels=ts.labels, score=round(corr, 3)
                    )
                )
        tmp_causes.sort(key=lambda x: x.score, reverse=True)
        return tmp_causes[:3]

    def get_container_extra_info(
        self, machine_id, container_name, start_time, end_time, obs_size
    ):
        result: Dict[str, Union[str, int, float]] = {
            "container_name": container_name,
            "machine_id": machine_id,
        }
        raw = self.config.extra_metrics or ""
        metrics = [m.strip() for m in raw.split(",") if m.strip()]
        for metric in metrics:
            ts_list = self.data_loader.get_metric(
                start_time, end_time, metric, machine_id=machine_id
            )
            for _ts in ts_list:
                if container_name == _ts.labels.get("container_name", ""):
                    trend = self.cal_trend(_ts.values, obs_size)
                    result[metric] = trend
                    break
        return result

    @staticmethod
    def _normalize_df(df: pd.DataFrame):
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.number):
                arr = df[col].to_numpy()
                mx, mn = float(np.max(arr)), float(np.min(arr))
                if mx != mn:
                    df[col] = (df[col] - mn) / (mx - mn)

    @staticmethod
    def _check_level(metric_data: np.ndarray, level: float) -> float:
        data_size = len(metric_data)
        if int(data_size * (1 - level)) == 0:
            peak = 2
            level = 1.0 - peak / float(data_size) - 1e-6
        return level

    @staticmethod
    def _is_data_valid(values, point_count, obs_size) -> bool:
        if len(values) < point_count * 0.6 or np.allclose(values, 0):
            return False
        if all(x == values[0] for x in values):
            return False
        return True

    @staticmethod
    def cal_trend(metric_values: List[float], obs_size: int) -> float:
        if len(metric_values) <= obs_size:
            return 0.0
        pre, check = metric_values[:-obs_size], metric_values[-obs_size:]
        pre_mean, check_mean = np.mean(pre), np.mean(check)
        return round((check_mean - pre_mean) / pre_mean, 3) if pre_mean > 0 else 0.0


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


@mcp.tool(name="container_disruption_detection_tool")
def container_disruption_detection_tool(
    kpis: List[KPIParam] = None,
    window: WindowParam = WindowParam(),
    extra: Optional[ExtraConfig] = None,
    anteater_conf: Optional[dict] = None,
    metric_info: Optional[dict] = None,
    machine_id: Optional[str] = None,
) -> List[AnomalyModel]:
    """容器异常检测工具（支持自动识别机器ID）"""
    job_path = os.path.join(os.path.dirname(__file__), "../config/container_disruption.job.json")
    anteater_conf = os.path.join(os.path.dirname(__file__), "../config/gala-anteater.yaml")
    kpis, window, extra = load_kpis_from_job(job_path)
    print(f"kpis: {kpis}, window: {window}, extra: {extra}")

    loader = build_metric_loader(config_path=anteater_conf, metricinfo_json=metric_info)
    facade = ContainerDisruptionFacade(loader, extra or ExtraConfig())
    anomalies: List[AnomalyModel] = []

    # 如果未指定 machine_id，则自动获取
    machine_ids: List[str] = []
    if not machine_id:
        # 选择第一个 KPI 的 metric 进行扫描
        if not kpis:
            raise ValueError("必须提供至少一个 KPI 参数")
        machine_ids = facade.get_unique_machine_ids(window.look_back, kpis)
    else:
        machine_ids = [machine_id]

    # 对每个机器执行检测
    for mid in machine_ids:
        for k in kpis:
            kth = float(k.params.get("outlier_ratio_th", 0.1))
            anomalies.extend(
                facade.detect_by_spot(
                    k.metric, mid, kth, window.look_back, window.obs_size
                )
            )

    logger.info("total containers: %d", facade.container_num)
    facade.container_num = 0
    return anomalies


@mcp.tool(name="rca_tool")
def rca_tool(
    metric: str,
    victim_container_name: str,
    window: WindowParam = WindowParam(),
    anteater_conf: Optional[dict] = None,
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

    # # 验证配置文件和检测逻辑是否可正常运行
    # job_path = os.path.join(os.path.dirname(__file__), "../config/container_disruption.job.json")
    # anteater_conf_path = os.path.join(os.path.dirname(__file__), "../config/gala-anteater.yaml")

    # kpis, window, extra = load_kpis_from_job(job_path)
    # logger.info("Container Disruption Detection MCP configuration loaded successfully.")
    # print(kpis, window, extra)

    # metric_info = {}

    # # 调用检测逻辑进行一次运行验证
    # anomalies = container_disruption_detection_tool(
    #     kpis=kpis,
    #     window=window,
    #     extra=extra,
    #     anteater_conf=anteater_conf_path,
    #     metric_info=metric_info,
    # )

    # ====== 启动 MCP 服务 ======
    mcp.run(transport="sse")
