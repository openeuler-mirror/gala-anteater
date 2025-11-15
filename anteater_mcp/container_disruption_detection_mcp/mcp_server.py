from __future__ import annotations
import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any

from common.loader import build_metric_loader
from mcp.server import FastMCP

from anteater.core.anomaly import Anomaly, RootCause
from anteater.core.ts import TimeSeries
from anteater.core.kpi import KPI, ModelConfig
from anteater.model.detector.disruption_detector import ContainerDisruptionDetector

from anteater_mcp.container_disruption_detection_mcp.mcp_data import (
    PerceptionResult,
    AnomalyInfo,
    KPIParam,
    ExtraConfig,
)

from anteater_mcp.container_disruption_detection_mcp.utils import (
    load_kpis_from_job,
    dt_last,
)

from anteater_mcp.container_disruption_detection_mcp.api.disruption_source_api import (
    DisruptionSourceAPI,
)


# -------------------------
# 日志配置
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("container_disruption_detection_mcp")

# -------------------------
# 启动 MCP Server
# -------------------------
mcp = FastMCP("Container Disruption Detection MCP", host="0.0.0.0", port=12345)


# -----------------------------------------------------
#  Facade：对 ContainerDisruptionDetector 做统一封装
# -----------------------------------------------------
class ContainerDisruptionFacade:
    """
    用于统一屏蔽 ContainerDisruptionDetector 的复杂接口，
    使 MCP 工具调用更稳定、更可控。
    """

    def __init__(self, data_loader, config: ModelConfig | ExtraConfig):
        self.data_loader = data_loader

        # 兼容 ExtraConfig / ModelConfig
        if isinstance(config, ModelConfig):
            self.detector = ContainerDisruptionDetector(data_loader, config)
        else:
            self.detector = ContainerDisruptionDetector(
                data_loader,
                ModelConfig(
                    name="container_disruption_detection",
                    model_path="../../anteater/model/detector/disruption_detector.py",
                    params={"extra_metrics": getattr(config, "extra_metrics", "")},
                ),
            )

    def get_unique_machine_id(self, look_back: int, kpis: List[KPI]):
        start, end = dt_last(minutes=look_back)
        mids = self.detector.get_unique_machine_id(start, end, kpis)
        return start, end, mids

    def detect_by_spot(self, kpi, machine_id: str) -> List[Anomaly]:
        return self.detector.detect_by_spot(kpi, machine_id)

    def get_kpi_ts_list(self, metric: str, machine_id: str, look_back: int):
        return self.detector.get_kpi_ts_list(metric, machine_id, look_back)

    def find_disruption_source(self, victim_ts: TimeSeries, all_ts: List[TimeSeries]):
        return self.detector.find_disruption_source(victim_ts, all_ts)

    def get_container_extra_info(
        self, machine_id, container_name, start_time, end_time, obs_size
    ):
        return self.detector.get_container_extra_info(
            machine_id, container_name, start_time, end_time, obs_size
        )


# -----------------------------------------------------
# 工具函数
# -----------------------------------------------------
def _score_to_level(score: float) -> str:
    """将 0~1 score 映射为异常级别"""
    if score < 0.3:
        return "轻度"
    elif score < 0.7:
        return "中度"
    return "严重"


def _extract_container_id(labels: Dict[str, Any]) -> str:
    """尽最大可能从 labels 中提取容器 ID"""
    if not labels:
        return ""
    for k in ("container_id", "container_name", "pod", "instance"):
        if k in labels and labels[k]:
            return str(labels[k])
    return str(labels)


def _build_detection_report(
    task_id: str,
    perception: PerceptionResult,
    kpis: List[KPIParam],
    start_ts_ms: int,
    end_ts_ms: int,
    container_keyword_list: List[str],
) -> Dict[str, Any]:

    metric_list = sorted({k.metric for k in kpis}) if kpis else []
    details = []

    for anomaly in perception.anomaly_info:
        cid = _extract_container_id(anomaly.labels)

        # 容器关键字过滤
        if container_keyword_list:
            if not any(kw in cid for kw in container_keyword_list):
                continue

        abnormal_level = _score_to_level(anomaly.score)

        # 峰值提取
        peak = None
        if isinstance(anomaly.details, dict):
            for key in ("peak", "max", "value"):
                if key in anomaly.details:
                    try:
                        peak = int(float(anomaly.details[key]))
                        break
                    except Exception:
                        pass
        if peak is None:
            peak = int(anomaly.score * 100)


        def _safe_parse_dt(ts):
            if isinstance(ts, datetime):
                return ts
            if isinstance(ts, str):
                try:
                    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except:
                    return None
            return None

        inner_info = anomaly.details.get("info", {}) if isinstance(anomaly.details, dict) else {}

        inner_start = _safe_parse_dt(inner_info.get("abnormal_start"))
        inner_end = _safe_parse_dt(inner_info.get("abnormal_end"))

        use_start = int(inner_start.timestamp() * 1000) if inner_start else start_ts_ms
        use_end = int(inner_end.timestamp() * 1000) if inner_end else end_ts_ms
        
        details.append(
            {
                "container_id": cid,
                "metric_id": anomaly.metric,
                "start_timestamp": use_start,
                "end_timestamp": use_end,
                "abnormal_level": abnormal_level,
                "disruption_peak": peak,
            }
        )

    return {
        "task_id": task_id,
        "code": 200,
        "msg": "success",
        "detection_report": {
            "metric_list": metric_list,
            "disruption_cnt": len(details),
            "details": details,
        },
    }


# -----------------------------------------------------
# MCP 1：容器干扰检测（新版接口）
# -----------------------------------------------------
@mcp.prompt(description="容器干扰检测 MCP：根据给定时间范围检测 SLI 是否存在干扰症状。")
@mcp.tool(name="container_disruption_detection_tool")
def container_disruption_detection_tool(request: str) -> Dict[str, Any]:
    """
    容器干扰检测 API（与文档对齐）：
    - 入参为 JSON 字符串 request
      {
        "task_id": "...",
        "container_keyword_list": [...],   // 可选
        "metric_keyword_list": [...],      // 可选
        "analysis_timestamp": 1234567890,  // 可选，毫秒
        "analysis_interval": 600000        // 必选，毫秒
      }
    - 输出 JSON（字典）
    """
    # -------------- 参数解析 --------------
    try:
        payload = json.loads(request)
    except Exception as e:
        logger.exception("请求 JSON 解析失败")
        return {"task_id": "", "code": 400, "msg": f"invalid json: {e}"}

    task_id = payload.get("task_id", "")
    if not task_id:
        return {"task_id": "", "code": 400, "msg": "task_id is required"}

    if "analysis_interval" not in payload:
        return {"task_id": task_id, "code": 400, "msg": "analysis_interval is required"}

    try:
        analysis_interval = int(payload["analysis_interval"])
    except Exception:
        return {"task_id": task_id, "code": 400, "msg": "analysis_interval must be int"}

    if analysis_interval <= 0:
        return {"task_id": task_id, "code": 400, "msg": "analysis_interval must > 0"}

    analysis_timestamp = payload.get("analysis_timestamp", int(time.time() * 1000))
    try:
        analysis_timestamp = int(analysis_timestamp)
    except Exception:
        return {"task_id": task_id, "code": 400, "msg": "analysis_timestamp must be int"}

    start_ts_ms = analysis_timestamp - analysis_interval
    end_ts_ms = analysis_timestamp
    look_back_minutes = max(1, int(analysis_interval / 60000))

    container_keywords = [str(x) for x in payload.get("container_keyword_list", [])]
    metric_keywords = [str(x) for x in payload.get("metric_keyword_list", [])]

    logger.info(
        "开始检测 | task_id=%s | look_back=%d min | metrics_kw=%s | containers_kw=%s",
        task_id,
        look_back_minutes,
        metric_keywords,
        container_keywords,
    )

    # ------------------ 加载 job 配置 ------------------
    job_path = os.path.join(
        os.path.dirname(__file__), "../config/container_disruption.job.json"
    )
    anteater_conf = os.path.join(
        os.path.dirname(__file__), "../config/gala-anteater.yaml"
    )

    try:
        kpis, window_cfg, extra_cfg = load_kpis_from_job(job_path)
    except Exception:
        logger.exception("加载 job 配置失败")
        return {"task_id": task_id, "code": 404, "msg": "failed to load job config"}

    # 根据 metric 关键字过滤 KPI
    if metric_keywords:
        kpis = [k for k in kpis if any(kw in k.metric for kw in metric_keywords)]

    # 无 KPI → 正常返回（空报告）
    if not kpis:
        logger.warning("无可分析的 KPI，直接返回空报告")
        empty = PerceptionResult(
            is_anomaly=False, anomaly_info=[], start_time=start_ts_ms, end_time=end_ts_ms
        )
        return _build_detection_report(
            task_id, empty, [], start_ts_ms, end_ts_ms, container_keywords
        )

    # ------------------ 构造检测器 ------------------
    try:
        loader = build_metric_loader(config_path=anteater_conf, metricinfo_json=None)
        facade = ContainerDisruptionFacade(
            loader,
            ModelConfig(
                name="container_disruption_detection",
                model_path="../../anteater/model/detector/disruption_detector.py",
                params={"extra_metrics": extra_cfg.extra_metrics},
            ),
        )
    except Exception:
        logger.exception("构造 detector 失败")
        return {"task_id": task_id, "code": 404, "msg": "failed to init detector"}

    # ------------------ 获取机器列表 ------------------
    try:
        start_dt, end_dt, machine_ids = facade.get_unique_machine_id(
            look_back_minutes, kpis
        )
    except Exception:
        logger.exception("获取机器列表失败")
        return {"task_id": task_id, "code": 404, "msg": "failed to query machine list"}

    if not machine_ids:
        return {"task_id": task_id, "code": 404, "msg": "no machine data available"}

    logger.info(
        "检测窗口 job=[%s - %s]  actual=[%s - %s]",
        start_dt.isoformat(),
        end_dt.isoformat(),
        datetime.fromtimestamp(start_ts_ms / 1000).isoformat(),
        datetime.fromtimestamp(end_ts_ms / 1000).isoformat(),
    )

    # --------------------------------------------------
    # 执行检测流程（遍历机器和 KPI）
    # --------------------------------------------------
    anomalies: List[Anomaly] = []
    perception = PerceptionResult(
        is_anomaly=False,
        anomaly_info=[],
        start_time=int(start_ts_ms / 1000),
        end_time=int(end_ts_ms / 1000),
    )

    try:
        for mid in machine_ids:
            for kpi in kpis:
                anomalies.extend(facade.detect_by_spot(kpi, mid))
    except Exception:
        logger.exception("检测执行失败")
        return {
            "task_id": task_id,
            "code": 404,
            "msg": "error occurred during detection",
        }

    logger.info(
        "检测结束 | task_id=%s | 总机器数=%d | 异常数=%d",
        task_id,
        len(machine_ids),
        len(anomalies),
    )

    # --------------------------------------------------
    # 转换为 PerceptionResult.AnomalyInfo
    # --------------------------------------------------
    anomaly_infos: List[AnomalyInfo] = []
    for an in anomalies:
        anomaly_infos.append(
            AnomalyInfo(
                machine_id=getattr(an, "machine_id", ""),
                metric=getattr(an, "metric", ""),
                labels=getattr(an, "labels", {}) or {},
                score=float(getattr(an, "score", 0.0)),
                entity_name=getattr(an, "entity_name", "") or "",
                details=getattr(an, "details", {}) or {},
            )
        )

    if anomaly_infos:
        perception.is_anomaly = True
        perception.anomaly_info = anomaly_infos

    # --------------------------------------------------
    # 构造对外 detection_report
    # --------------------------------------------------
    result = _build_detection_report(
        task_id,
        perception,
        kpis,
        start_ts_ms,
        end_ts_ms,
        container_keywords,
    )

    logger.info(
        "检测报告完成 | task_id=%s | disruption_cnt=%d",
        task_id,
        result["detection_report"]["disruption_cnt"],
    )
    return result


# =========================================================
# MCP 2：容器干扰源分析
# =========================================================
@mcp.prompt(description="容器干扰源分析工具：基于检测报告分析可能的干扰源。")
@mcp.tool(name="container_interference_analysis_tool")
def container_interference_analysis_tool(request: str) -> Dict[str, Any]:
    """
    干扰源分析 API：
    入参为 JSON 字符串：
    {
      "task_id": "...",
      "detection_report": { ... }   // 容器干扰检测工具输出的检测报告
    }
    """
    # ---------- 解析请求 ----------
    try:
        payload = json.loads(request)
    except Exception as e:
        logger.exception("[Analysis] invalid json")
        return {"task_id": "", "code": 400, "msg": f"invalid json: {e}"}

    task_id = payload.get("task_id", "")
    if not task_id:
        return {"task_id": "", "code": 400, "msg": "task_id is required"}

    detection_report = payload.get("detection_report")
    if not isinstance(detection_report, dict):
        return {
            "task_id": task_id,
            "code": 400,
            "msg": "detection_report is required and must be dict",
        }

    logger.info("[Analysis] start | task_id=%s", task_id)

    # ---------- 校验 detection_report ----------
    if "details" not in detection_report:
        return {
            "task_id": task_id,
            "code": 400,
            "msg": "invalid detection_report: missing details",
        }

    if detection_report.get("disruption_cnt", 0) == 0:
        # 无异常 → 可直接返回
        return {
            "task_id": task_id,
            "code": 200,
            "msg": "no disruption detected",
            "analysis_report": {
                "metric_list": detection_report.get("metric_list", []),
                "disruption_cnt": 0,
                "details": [],
            },
        }

    details = detection_report.get("details") or []
    if not details:
        return {
            "task_id": task_id,
            "code": 200,
            "msg": "no disruption detected",
            "analysis_report": {
                "metric_list": detection_report.get("metric_list", []),
                "disruption_cnt": 0,
                "details": [],
            },
        }

    d = details[0]
    victim_container_id = d.get("container_id")
    metric_id = d.get("metric_id")

    if not victim_container_id or not metric_id:
        return {
            "task_id": task_id,
            "code": 400,
            "msg": "invalid detection_report: container_id/metric_id missing",
        }

    # NOTE：检测报告没有 machine_id，因此从 metric_loader 再扫描一次
    anteater_conf = os.path.join(
        os.path.dirname(__file__), "../config/gala-anteater.yaml"
    )

    try:
        loader = build_metric_loader(config_path=anteater_conf, metricinfo_json={})
    except Exception as e:
        logger.exception("[Analysis] metric loader failure")
        return {
            "task_id": task_id,
            "code": 404,
            "msg": f"metric loader failure: {e}",
        }

    # ---------- 获取 SLI 时间序列 ----------
    try:
        start_dt, end_dt = dt_last(minutes=2)
        all_ts: List[TimeSeries] = loader.get_metric(start_dt, end_dt, metric_id)
    except Exception:
        logger.exception("[Analysis] metric data fetch failed")
        return {
            "task_id": task_id,
            "code": 404,
            "msg": "metric data fetch failed",
        }

    if not all_ts:
        return {
            "task_id": task_id,
            "code": 404,
            "msg": "no metric ts found",
        }

    # ---------- 找受害者 ts ----------
    victim_ts = None
    for ts in all_ts:
        if ts.labels.get("container_id") == victim_container_id:
            victim_ts = ts
            break

    if victim_ts is None:
        return {
            "task_id": task_id,
            "code": 404,
            "msg": "victim ts not found",
        }

    # ---------- 调用 DisruptionSourceAPI ----------
    try:
        api = DisruptionSourceAPI()
        sources: List[RootCause] = api.find_sources(victim_ts, all_ts)
    except Exception as e:
        logger.exception("[Analysis] 干扰源计算失败")
        # 文档只规定 200/400/404，这里归为数据访问类错误
        return {
            "task_id": task_id,
            "code": 404,
            "msg": f"find_sources error: {e}",
        }

    # ---------- 概率归一化 ----------
    probs: Dict[str, float] = {}
    total = sum(float(getattr(s, "score", 0.0)) for s in sources)
    for rc in sources:
        cid = rc.labels.get("container_id", "")
        if not cid:
            continue
        probs[cid] = round(
            (float(getattr(rc, "score", 0.0)) / total), 3
        ) if total > 0 else 0.0

    # ---------- 构造输出 ----------
    result = {
        "task_id": task_id,
        "code": 200,
        "msg": "success",
        "analysis_report": {
            "metric_list": [metric_id],
            "disruption_cnt": len(sources),
            "details": [
                {
                    "container_id": victim_container_id,
                    "disrupted_metric_id": metric_id,
                    "interf_src_probs": probs,
                }
            ],
        },
    }

    logger.info(
        "[Analysis] completed | task_id=%s | root_causes=%d",
        task_id,
        len(sources),
    )
    return result


# =========================================================
# MCP 3：容器干扰恢复建议
# =========================================================
@mcp.prompt(description="容器干扰恢复建议工具：基于检测 + 分析结果生成恢复建议。")
@mcp.tool(name="container_interference_recovery_suggestion_tool")
def container_interference_recovery_suggestion_tool(request: str) -> Dict[str, Any]:
    """
    干扰恢复建议 API：
    入参为 JSON 字符串：
    {
      "task_id": "...",
      "detection_report": { ... },   // 检测工具输出
      "analysis_report": { ... }     // 分析工具输出
    }
    """
    # ---------- 解析请求 ----------
    try:
        payload = json.loads(request)
    except Exception as e:
        logger.exception("[Recovery] invalid json")
        return {"task_id": "", "code": 400, "msg": f"invalid json: {e}"}

    task_id = payload.get("task_id", "")
    if not task_id:
        return {"task_id": "", "code": 400, "msg": "task_id is required"}

    detection_report = payload.get("detection_report")
    analysis_report = payload.get("analysis_report")

    if not isinstance(detection_report, dict):
        return {
            "task_id": task_id,
            "code": 400,
            "msg": "detection_report is required and must be dict",
        }

    if not isinstance(analysis_report, dict):
        return {
            "task_id": task_id,
            "code": 400,
            "msg": "analysis_report is required and must be dict",
        }

    logger.info("[Recovery] start | task_id=%s", task_id)

    # 参数校验（analysis_report）
    details = analysis_report.get("details", [])
    if not isinstance(details, list):
        return {
            "task_id": task_id,
            "code": 400,
            "msg": "invalid analysis_report: details must be list",
        }

    if not details:
        return {
            "task_id": task_id,
            "code": 200,
            "msg": "no interference found",
            "recovery_suggestion": [],
        }

    d = details[0]
    victim = d.get("container_id")
    metric = d.get("disrupted_metric_id")
    probs = d.get("interf_src_probs", {})

    if not victim or not metric:
        return {
            "task_id": task_id,
            "code": 400,
            "msg": "invalid analysis_report: container_id/disrupted_metric_id missing",
        }

    # --------- 没有明确干扰源 ----------
    if not probs:
        return {
            "task_id": task_id,
            "code": 200,
            "msg": "success",
            "recovery_suggestion": [
                {
                    "suggestion": f"容器 {victim} 未检测到明确干扰源，可继续观察。",
                    "evidence": "干扰源概率均为 0。",
                    "example": "无需操作。",
                }
            ],
        }

    # 选最可能的干扰源
    src_id, prob = max(probs.items(), key=lambda x: x[1])

    suggestion = (
        f"容器 {victim} 的指标 {metric} 受到干扰，"
        f"建议对可能的干扰源容器 {src_id} 进行性能隔离或限制资源。"
    )
    evidence = f"干扰源概率最高的容器为 {src_id}（概率 {prob}）。"
    example = (
        f"示例操作：为容器 {src_id} 设置更严格的资源限制，"
        f"或将容器 {victim} 调度到独占节点运行。"
    )

    result = {
        "task_id": task_id,
        "code": 200,
        "msg": "success",
        "recovery_suggestion": [
            {"suggestion": suggestion, "evidence": evidence, "example": example}
        ],
    }

    logger.info("[Recovery] finish | task_id=%s", task_id)
    return result


# =========================================================
# MCP Server 主入口
# =========================================================
if __name__ == "__main__":
    if os.name == "posix":
        import multiprocessing

        multiprocessing.set_start_method("spawn", force=True)

    logger.info("启动 MCP Server (SSE 模式)，端口=12345")
    mcp.run(transport="sse")
