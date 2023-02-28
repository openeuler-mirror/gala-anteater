import warnings

import os
from collections import defaultdict
from functools import reduce
from os import path, sep
from typing import List, Dict

import numpy as np
import pandas as pd

from anteater.config import AnteaterConf
from anteater.core.time_series import TimeSeries
from anteater.factory.factory import DataCollectorFactory
from anteater.utils import datetime
from anteater.utils.common import divide
from anteater.utils.log import logger
from anteater.utils.timer import timer


def load_datasets():
    folder_path = path.dirname(path.dirname(path.realpath(__file__)))
    data_folder = path.join(folder_path, sep.join(['data', 'sample']))

    time_series_set = []
    for machine in os.listdir(data_folder):
        data_path = os.path.join(data_folder, machine)
        for root, _, file_names in os.walk(data_path):
            for file_name in file_names:
                file_path = os.path.join(root, file_name)
                time_stamps, values = parse_csv(file_path)
                metric, labels = parse_file_name(file_name)
                labels['machine_id'] = machine

                time_series_set.append(TimeSeries(
                    metric=metric,
                    labels=labels,
                    time_stamps=time_stamps,
                    values=values))
    logger.info("Load file completed, start to combine time series data!")

    return combine_ts(time_series_set)


def parse_csv(file_path: str):
    if file_path.startswith(u"\\\\"):
        file_path = os.sep * 2 + '?' + os.sep + 'UNC' + os.sep + file_path[2:]
    else:
        file_path = os.sep * 2 + '?' + os.sep + file_path

    if not path.isfile(file_path):
        raise FileNotFoundError(f'File {file_path} is not found')
    data = pd.read_csv(file_path)
    return data['timestamps'].tolist(), data['values'].tolist()


def parse_file_name(file_name: str):
    idx1 = file_name.find('@')
    idx2 = file_name.rfind('.')

    metric = file_name[:idx1]
    labels = {}

    label_txt = file_name[idx1+1: idx2]
    for sub_str in label_txt.split('@'):
        i = sub_str.find('=')

        if i < 0:
            logger.error(f'Filename {file_name} parse failed!')
            continue

        key = sub_str[:i]
        val = sub_str[i+1:]

        if key in labels:
            raise ValueError(f'key {key} has already existed!')

        labels[key] = val

    return metric, labels


def combine_ts(time_series_set: List[TimeSeries]) -> Dict[str, List[TimeSeries]]:
    metric_map = defaultdict(list)

    for i, ts in enumerate(time_series_set):
        if i % 1000 == 0:
            logger.info(f'combine_ts {i}!')
        metric = ts.metric

        exist = False
        for metric_ts in metric_map.get(metric):
            if ts == metric_ts:
                metric_ts.insert(ts)
                exist = True
                break

        if not exist:
            metric_map[metric].append(ts)
    logger.info("Combine time series completed!")

    return metric_map


class TestMetricLoader:
    def __init__(self, config: AnteaterConf) -> None:
        self.datasets = load_datasets()
        self.provider = DataCollectorFactory.\
            get_instance(config.global_conf.data_source, config)

    @staticmethod
    def ts_operate(ts_list, key):
        warnings.simplefilter(action='ignore', category=FutureWarning)

        merged_datasets = []
        unique_values = set()
        for ts in ts_list:
            if ts.labels[key] not in unique_values:
                unique_values.add(ts.labels[key])

        for val in unique_values:
            tmp_ts_list = []
            for ts in ts_list:
                if ts.labels[key] == val:
                    tmp_ts_list.append(ts)

            if len(tmp_ts_list) == 1:
                merged_datasets.append(tmp_ts_list[0])

            else:
                metric_dfs = ([ts.to_df(str(i)) for i, ts in enumerate(tmp_ts_list)])
                df = reduce(lambda left, right: pd.DataFrame(left).join(right, how='outer'), metric_dfs)
                df['mean'] = df.apply(lambda x: np.average([v for v in x if not pd.isna(v)]), axis=1)
                time_stamps = [int(x) for x in divide(df.index.astype(int), 10**9)]
                values = df['mean'].tolist()

                merged_datasets.append(
                    TimeSeries(
                        metric=tmp_ts_list[0].metric,
                        labels={key: val},
                        time_stamps=time_stamps,
                        values=values)
                )

        return merged_datasets

    def get_metric(self, start: datetime, end: datetime, metric: str, **kwargs) -> List[TimeSeries]:
        if metric not in self.datasets:
            logger.warning(f'TestMetricLoader: empty metric {metric} in datasets!')
            return []

        start, end = round(start.timestamp()), round(end.timestamp())
        datasets = self.datasets.get(metric)
        if 'machine_id' in kwargs:
            machine_id = kwargs.get('machine_id')
            datasets = [ts for ts in datasets if ts.labels['machine_id'] == machine_id]

        filtered_datasets = []
        for ts in datasets:
            indexes = [i for i, v in enumerate(ts.time_stamps) if start <= v <= end]
            time_stamps = [ts.time_stamps[i] for i in indexes]
            values = [ts.values[i] for i in indexes]

            filtered_datasets.append(
                TimeSeries(
                    metric=ts.metric,
                    labels=ts.labels,
                    time_stamps=time_stamps,
                    values=values))

        if 'operator' in kwargs and len(filtered_datasets) > 1:
            if kwargs.get('operator') != 'avg':
                raise ValueError(f"Notimplemented operator {kwargs.get('operator')}!")

            return self.ts_operate(filtered_datasets, kwargs.get('keys'))

        else:
            return filtered_datasets

    @timer
    def get_unique_machines(self, start: datetime, end: datetime, metrics: List[str]) -> List[str]:
        machine_ids = self.get_unique_label(start, end, metrics, label_name="machine_id")
        if not machine_ids:
            logger.warning(f'Empty unique machine ids on given metrics!')
            logger.debug(f'metrics: {";".join(metrics)}')
        else:
            logger.info(f'Got {len(machine_ids)} unique machine_ids on given metrics')
            logger.debug(f'machine ids: {";".join(machine_ids)}')
            logger.debug(f'metrics: {";".join(metrics)}')
        return machine_ids

    def get_unique_label(self, start: datetime, end: datetime, metrics: List[str], label_name: str) -> List[str]:
        unique_labels = set()
        for metric in metrics:
            time_series = self.get_metric(start, end, metric)
            unique_labels.update([item.labels.get(label_name, "") for item in time_series])

        return list([lbl for lbl in unique_labels if lbl])

    def expected_point_length(self, start: datetime, end: datetime) -> int:
        start, end = round(start.timestamp()), round(end.timestamp())
        if self.provider.step >= 0:
            return max((end - start) // self.provider.step, 1)
        else:
            return 0
