from __future__ import annotations

import json
import logging

from typing import List, Tuple
from datetime import datetime, timedelta, timezone

from anteater_mcp.container_disruption_detection_mcp.mcp_data import (
    KPIParam,
    WindowParam,
    ExtraConfig,
)

logger = logging.getLogger("container_disruption_detection_mcp.utils")


# 加载 job.json，解析为 KPIParam / WindowParam / ExtraConfig
def load_kpis_from_job(
    job_path: str, look_back_minutes: int = 20
) -> Tuple[List[KPIParam], WindowParam, ExtraConfig]:
    """
    加载 container_disruption.job.json
    输出：
        - KPIParam 列表
        - WindowParam（look_back，obs_size）
        - ExtraConfig（extra_metrics）
    """

    logger.info(f"[load_kpis_from_job] 加载 job 文件: {job_path}")

    try:
        with open(job_path, "r", encoding="utf-8") as f:
            job = json.load(f)
    except Exception as e:
        logger.exception(f"[load_kpis_from_job] 读取 job.json 失败: {e}")
        raise

    # 解析 KPIParam
    kpis: List[KPIParam] = []

    for idx, k in enumerate(job.get("kpis", [])):
        if not k.get("enable", True):
            continue

        metric = k.get("metric")
        if not metric:
            logger.warning(f"[load_kpis_from_job] KPI[{idx}] 缺少 metric，跳过")
            continue
        entity_name = str(k.get("entity_name", ""))
        params = k.get("params", {}) or {}
        params.update({"look_back": look_back_minutes})
        kp = KPIParam(
            metric=metric,
            entity_name=entity_name,
            params=params,
        )
        kpis.append(kp)

    logger.info(f"[load_kpis_from_job] 已加载 {len(kpis)} 个 KPIParam")

    # 解析 WindowParam
    if kpis:
        first_params = kpis[0].params
        look_back = look_back_minutes if look_back_minutes > 0 else int(first_params.get("look_back", 20))
        obs_size = int(first_params.get("obs_size", 20))
    else:
        look_back, obs_size = look_back_minutes if look_back_minutes > 0 else 20, 20  # 默认值

    window = WindowParam(look_back=look_back, obs_size=obs_size)

    logger.info(
        f"[load_kpis_from_job] WindowParam: look_back={look_back}, obs_size={obs_size}"
    )

    # 解析 ExtraConfig
    model_conf = job.get("model_config", {}).get("params", {})
    extra_metrics = str(model_conf.get("extra_metrics", "")).strip()
    extra = ExtraConfig(extra_metrics=extra_metrics)

    logger.info(
        f"[load_kpis_from_job] ExtraMetrics='{extra_metrics if extra_metrics else '(none)'}'"
    )

    return kpis, window, extra


# 回溯时间窗口
def dt_last(*, minutes: int):
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes)
    return start, end
