from datetime import datetime
from functools import reduce

import pandas as pd
import numpy as np

from anteater.model.rca.data_load.metric_loader import MetricLoader
from anteater.model.rca.data_load.prometheus import load_prometheus_client
from anteater.model.rca.data_load.pearson import pearson_correlation, select_relevant_kpi
from anteater.model.rca.data_load.time_series import TimeSeries


def get_data_by_key(self, start, end, metric: str, key: str, label_id: str):
    if key == "machine_id":
        _ts_list = self.data_loader.get_metric(start, end, metric, operator='avg',
                        keys="machine_id", machine_id=label_id)
    elif key == "pod_id":
        _ts_list = self.data_loader.get_metric(start, end, metric, operator='avg',
                        keys="pod_id", pod_id=label_id)
    else:
        raise ValueError(
            f'The key-[{key}] only support machine_id or pod_id.')

    return _ts_list


def get_dataframe(start, end, kpis, machine_id, data_loader, keys=None):
    """Gets the features during a period seperated by machine ids"""
    metrics = kpis
    entity_id = machine_id
    
    if "." in entity_id:
        key = 'machine_id'
    else:
        key = 'pod_id'

    start_timestamp = datetime.timestamp(start)
    end_timestamp = datetime.timestamp(end)
    timestamp_list = np.arange(start_timestamp, end_timestamp + 5, 5)
    datetime_list = [datetime.strftime(datetime.utcfromtimestamp(timestamp), "%Y-%m-%d %H:%M:%S")
                     for timestamp in timestamp_list]
    timestamp = pd.to_datetime(np.asarray(timestamp_list).astype(float) * 1000, unit="ms")
    index = pd.to_datetime(timestamp)
    series = pd.Series(datetime_list, index=index, name="timestamp")
    series = series[~series.index.duplicated()]


    metric_dfs = []
    metrics_list = []
    
    # extend machine_id -> machine_id, pod_id
    for metric in metrics:
        if key == 'machine_id':
            if 'container' in metric:
                continue
            _ts_list = data_loader.get_metric(start, end, metric, operator='avg', keys="machine_id", machine_id=entity_id)
        else: 
            _ts_list = data_loader.get_metric(start, end, metric, operator='avg', keys="pod_id", pod_id=entity_id)

        if _ts_list:
            _ts_list = _ts_list[0]
        else:
            _ts_list = TimeSeries(metric, {}, [], [])

        if _ts_list.to_df().columns[0] not in metrics_list:
            metrics_list.append(_ts_list.to_df().columns[0])
            metric_dfs.append(_ts_list.to_df())

    df = reduce(lambda left, right: pd.DataFrame(
        left).join(right, how='outer'), metric_dfs)

    # 这块填值逻辑建议用平均值或者前后相邻的值代替
    # method = 'bflii'/'backfill'：用下一个非缺失值填充该缺失值
    # method = 'ffill'/'pad'：用前一个非缺失值去填充该缺失值
    # df = df.fillna(method='ffill') # 前一个非缺失值填充
    df = df.fillna(0)
    df["timestamp"] = df.index.to_list()
    
    return df


def get_single_dataframe(_single_ts, single_metrics_list, single_metric_dfs):
    if _single_ts.to_df().columns[0] not in single_metrics_list:
        single_metrics_list.append(_single_ts.to_df().columns[0])
        single_metric_dfs.append(_single_ts.to_df())
    return single_metric_dfs, single_metrics_list


def get_cause_dataframe(start_timestamp, end_timestamp, machine_id, metric, data_loader, data_persist=False):
    
    start = datetime.fromtimestamp(start_timestamp)
    end = datetime.fromtimestamp(end_timestamp)

    timestamp_list = np.arange(start_timestamp, end_timestamp + 5, 5)

    datetime_list = [datetime.strftime(datetime.utcfromtimestamp(timestamp), "%Y-%m-%d %H:%M:%S")
                     for timestamp in timestamp_list]
    timestamp = pd.to_datetime(np.asarray(timestamp_list).astype(float) * 1000, unit="ms")
    index = pd.to_datetime(timestamp)
    series = pd.Series(datetime_list, index=index, name="timestamp")
    series = series[~series.index.duplicated()]

    single_metric_dfs = []
    single_metrics_list = []

    single_metric_dfs.append(series.to_frame())

    
    _single_ts_list = data_loader.get_single_metric(
        start, end, metric, label_name=["machine_id"], label_value=[machine_id])

    single_df = pd.DataFrame({})
    if len(_single_ts_list):
        for _single_ts in _single_ts_list:
            single_metric_dfs, single_metrics_list = get_single_dataframe(
                _single_ts, single_metrics_list, single_metric_dfs)
        single_df = reduce(lambda left, right: pd.DataFrame(left).join(right, how='outer'), single_metric_dfs)
        single_df = single_df.fillna(0)

    return single_df


# def get_data(start, end, kpis, machine_id, front_end_metric):
#     client = load_prometheus_client()
#     data_loader = MetricLoader(client)
#     df = get_dataframe(start, end, kpis, machine_id, data_loader)
#     metric_records = {}
#     pearson_records = select_relevant_kpi(df, df[front_end_metric.split('@')[0]])
#     return metric_records, df, pearson_records


def get_data(start, end, kpis, machine_id):
    client = load_prometheus_client()
    data_loader = MetricLoader(client)
    df = get_dataframe(start, end, kpis, machine_id, data_loader)

    return df
