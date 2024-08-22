import numpy as np
from datetime import datetime, timezone
import os
import pandas as pd
from scipy.stats import pearsonr
from anteater.model.rca.data_load.get_data import get_cause_dataframe
from anteater.model.rca.data_load.metric_loader import MetricLoader
from anteater.model.rca.data_load.prometheus import load_prometheus_client
from anteater.utils.log import logger

import networkx as nx


def filter_metric_not_candidates(metric_candidates, metric: pd.DataFrame) -> pd.DataFrame:
    """
    根据候选指标名单对所有指标进行过滤
    """
    metrics_to_eliminate = []
    for column in metric.columns:
        if column not in metric_candidates + ["timestamp"]:
            metrics_to_eliminate.append(column)
    return metric.drop(columns=metrics_to_eliminate)


def _extrac_front_end_metric(resource_metric):
    front_end_metric = resource_metric
    if "@" in resource_metric:
        front_end_metric = resource_metric.split('@')[0]

    return front_end_metric


def search_entity_by_label(entities, labels):
    now_max = -1
    now_id = ""
    for entity in entities:
        max_num = 0
        for k in entity.raw_data.keys():
            if k in list(labels.keys()) and entity.raw_data[k] == labels[k]:
                max_num += 1
        if max_num > now_max:
            now_max = max_num
            now_id = entity.id
    return now_id


def date_to_timestamp(dt_str):
    """
    将"%Y-%m-%d %H:%M:%S"的utc时间字符串转为时间戳.

    Args:
        dt_str:  utc时间字符串

    Returns:
        时间戳(以s为单位)
    """
    if isinstance(dt_str, str):
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    else:
        dt = dt_str
    return dt.replace(tzinfo=timezone.utc).timestamp()


def smooth_data(df, window=18):
    """
    对dataframe数据使用滑动窗口进行平滑(不改变原有数据)

    Args:
        df: 需要处理的dataframe数据
        window: 滑动窗口大小,默认18

    Returns:
        平滑后的dataframe数据
    """
    for col in df.columns:
        if col == "timestamp":
            continue
        df[col] = df[col].rolling(window=window).mean().bfill().ffill().values
    return df


def search_similar_metric(reference_series: pd.Series, key_metrics: list, candidates_dir: str,
                          time_interval: tuple, smooth_window: int):
    start, end = time_interval

    metric_record = []
    for file in os.listdir(candidates_dir):
        file_path = os.path.join(candidates_dir, file)
        if not os.path.isfile(file_path):
            continue
        if file.split("@")[0] not in key_metrics:
            continue
        df = pd.read_csv(file_path)
        df = smooth_data(df, smooth_window)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).apply(lambda x: x.value // int(1e9))  # 时间戳单位为s
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
        if df.empty or len(df) != len(reference_series):
            continue
        df = df.reset_index(drop=True)
        correlation_coef = pearsonr(reference_series, df.loc[:, "value"])[0]
        if np.isnan(correlation_coef):
            continue
        metric_record.append((file.replace(".csv", ""), correlation_coef))
    metric_record = sorted(metric_record, key=lambda x: -x[1])
    logger.debug(f"metric_record: {metric_record}")
    return metric_record


def search_similar_metric(reference_series: pd.Series, key_metrics: list, candidates_dir: str,
                          time_interval: tuple, smooth_window: int, machine_id):
    start, end = time_interval

    client = load_prometheus_client()
    data_loader = MetricLoader(client)

    metric_record = []
    if "." in machine_id:
        key = 'machine_id'
    else:
        key = 'pod_id'

    for key_metric in key_metrics:
        if key == 'machine_id' and 'container' in key_metric:
            continue

        df = get_cause_dataframe(start, end, machine_id, key_metric, data_loader)

        if "timestamp" not in df.columns:
            continue
        df = smooth_data(df, smooth_window)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).apply(lambda x: x.value // int(1e9))  # 时间戳单位为s
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
        if df.empty or len(df) != len(reference_series):
            continue
        df = df.reset_index(drop=True)

        relevant_degree = []
        for col in df.columns:
            if col == "timestamp":
                continue
            correlation_coef = pearsonr(reference_series, df[col])[0]
            if np.isnan(correlation_coef):
                continue
            relevant_degree.append((col, correlation_coef))
        relevant_degree.sort(key=lambda x: -x[1])

        if len(relevant_degree) == 0:
            continue
        metric_record.append(relevant_degree[0])
    metric_record = sorted(metric_record, key=lambda x: -x[1])
    logger.info(f"metric_record: {metric_record}")
    return metric_record

