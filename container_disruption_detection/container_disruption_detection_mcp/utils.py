import json
import logging
from typing import Any, Dict, List, Tuple

from .mcp_data import KPIParam, WindowParam, ExtraConfig
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("container_disruption_detection_mcp")


def load_kpis_from_job(
    job_path: str,
) -> Tuple[List[KPIParam], WindowParam, ExtraConfig]:
    """从 job.json 中加载 KPIParam, WindowParam 和 ExtraConfig"""
    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)

    kpis = []
    for k in job.get("kpis", []):
        if not k.get("enable", True):
            continue
        kpis.append(
            KPIParam(
                metric=k["metric"],
                entity_name=k.get("entity_name", ""),
                params=k.get("params", {}),
            )
        )

    # 解析 WindowParam
    first_params = kpis[0].params if kpis else {}
    look_back = int(first_params.get("look_back", 20))
    obs_size = int(first_params.get("obs_size", 6))
    window = WindowParam(look_back=look_back, obs_size=obs_size)

    # 解析 ExtraConfig
    model_conf = job.get("model_config", {}).get("params", {})
    extra_metrics = model_conf.get("extra_metrics", "")
    extra = ExtraConfig(extra_metrics=extra_metrics)

    logger.info(
        f"Loaded {len(kpis)} KPI(s) from job file '{job_path}', "
        f"look_back={look_back}, obs_size={obs_size}, extra_metrics='{extra_metrics}'"
    )
    return kpis, window, extra


def dt_last(*, minutes: int):
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes)
    return start, end
