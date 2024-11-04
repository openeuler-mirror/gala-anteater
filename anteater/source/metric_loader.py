#!/usr/bin/python3
# ******************************************************************************
# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# gala-anteater is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# ******************************************************************************/
"""
Time:
Author:
Description: The implementation of metrics data loader.
"""

from datetime import datetime
from typing import List, Optional, Union

from anteater.config import AnteaterConf
from anteater.core.info import MetricInfo
from anteater.factory.clients import DataClientFactory
from anteater.utils.log import logger
from anteater.core.ts import TimeSeries
from anteater.utils.timer import timer


class MetricLoader:
    """
    The metric loader that consumes raw data from PrometheusAdapter,
    then convert them to dataframe
    """

    def __init__(self, metricinfo: MetricInfo, config: AnteaterConf) -> None:
        """The Metrics Loader initializer"""
        self.provider = DataClientFactory. \
            get_client(config.global_conf.data_source, config)

        self.metricinfo = metricinfo

    def get_single_metric(self, start: datetime, end: datetime, metric: str, **kwargs) -> List[TimeSeries]:
        """Get target metric time series data with attributes"""
        query = self._get_query(metric, **kwargs)
        time_series = self.provider.range_query(start, end, metric, query, is_single=True)

        return time_series

    def check_and_fill_data(self, ts_list, start, end):
        """Check whether the number of data records is normal."""
        start, end = round(start.timestamp()), round(end.timestamp())
        standard_time = list(range(start, end + self.provider.step, self.provider.step))
        # 将列表 data 转换为字典，方便查找和补充缺失的元素
        for i, ts_data in enumerate(ts_list):
            values = ts_data.values
            time_stamps = ts_data.time_stamps
            value_dict = dict(zip(time_stamps, values))
            if time_stamps != standard_time:
                # time_stamps = standard_time
                # 初始化结果列表
                result = []

                for index, time in enumerate(standard_time):
                    if time in value_dict:
                        # 如果 container_data 中存在相应的元素，直接添加到结果列表
                        result.append(value_dict[time])
                    else:
                        # 如果 container_data 中不存在相应的元素，进行补充
                        if index == 0:
                            # 如果是第一个元素缺失，用后一个元素填充
                            next_value = values[0]
                            result.append(next_value)
                        else:
                            # 如果不是第一个元素缺失，用前一个元素填充
                            next_value = result[-1]
                            result.append(next_value)
                ts_list[i].time_stamps = standard_time
                ts_list[i].values = result

        return ts_list

    def get_metric(self, start: datetime, end: datetime, metric: str, **kwargs) -> List[TimeSeries]:
        """Get target metric time series data

        :kwargs
            - label_name: Union[str, List] = None,
            - label_value: Union[str, List] = None,
            - operator_name: str = None,
            - operator_value: float = None)

        :return List of TimeSeries
        """
        query = self._get_query(metric, **kwargs)
        time_series = self.provider.range_query(start, end, metric, query)
        time_series = self.check_and_fill_data(time_series, start, end)
        return time_series

    @timer
    def get_topo(self, start: datetime, end: datetime, metrics: List[str], label_name: str = 'device_id') -> dict:
        """Gets unique labels of all metrics"""
        unique_labels = set()
        for metric in metrics:
            time_series = self.get_metric(start, end, metric)
            unique_labels.update(
                [(item.labels.get(label_name, ""), item.labels.get('instance', "")) for item in time_series])
        topo_info = dict()
        for lbl in unique_labels:
            pod, machine = lbl
            if machine not in topo_info.keys():
                pod_list = [pod]
                topo_info[machine] = pod_list
            else:
                topo_info[machine].append(pod)

        return topo_info

    @timer
    def get_unique_machines(self, start: datetime, end: datetime, metrics: List[str]) -> List[str]:
        """Gets the unique machine ids based on the metrics"""
        machine_ids = self.get_unique_label(start, end, metrics, label_name="machine_id")
        if not machine_ids:
            logger.warning('Empty unique machine ids on given metrics!')
            logger.debug(f'metrics: {";".join(metrics)}')
        else:
            logger.info(f'Got {len(machine_ids)} unique machine_ids on given metrics')
            logger.debug(f'machine ids: {";".join(machine_ids)}')
            logger.debug(f'metrics: {";".join(metrics)}')
        return machine_ids

    @timer
    def get_unique_pods(self, start: datetime, end: datetime, metrics: List[str]) -> List[str]:
        """Gets the unique machine ids based on the metrics"""
        pod_ids = self.get_unique_label(start, end, metrics, label_name="pod_id")
        if not pod_ids:
            logger.warning(f'Empty unique pod ids on given metrics!')
            logger.debug(f'metrics: {";".join(metrics)}')
        else:
            logger.info(f'Got {len(pod_ids)} unique pod_ids on given metrics')
            logger.debug(f'pod ids: {";".join(pod_ids)}')
            logger.debug(f'metrics: {";".join(metrics)}')

        return pod_ids

    def get_unique_label(self, start: datetime, end: datetime, metrics: List[str], label_name: str) -> List[str]:
        """Gets unique labels of all metrics"""
        unique_labels = set()
        for metric in metrics:
            time_series = self.get_metric(start, end, metric)
            unique_labels.update([item.labels.get(label_name, "") for item in time_series])

        return list([lbl for lbl in unique_labels if lbl])

    def expected_point_length(self, start: datetime, end: datetime) -> int:
        """Gets expected length of time series during a period"""
        start, end = round(start.timestamp()), round(end.timestamp())
        if self.provider.step > 0:
            return (end - start) // self.provider.step + 1
        else:
            return 0

    @staticmethod
    def _get_query(metric: str, operator: Optional[str] = None,
                   value: Optional[float] = None,
                   keys: Optional[Union[str, List]] = None, **labels):
        """Gets aggregated query patterns

        Format: [operator]([value,] metric{[**labels]}) by (keys)

            Such as:
                - 1. gala_gopher_bind_sends{machine_id="1234"}
                - 2. sum(gala_gopher_bind_sends) by (machine_id)
                - 2. sum(gala_gopher_bind_sends) by (machine_id)
                - 3. sum(gala_gopher_bind_sends{machine_id="1234"}) by (machine_id)
                - 4. quantile(0.7, gala_gopher_bind_sends{machine_id="1234"}) by (machine_id)
        """
        if operator and not keys:
            raise ValueError("Please provide param 'keys' when specified 'operator'!")

        rule = ""
        if labels:
            pairs = ",".join([f"{n}=~'{v}'" for n, v in labels.items()])
            rule = f"{{{pairs}}}"

        group = ""
        if isinstance(keys, list):
            group = ",".join([k for k in keys])
        elif isinstance(keys, str):
            group = keys

        if operator and value:
            query = f"{operator}({value}, {metric}{rule}) by ({group})"
        elif operator:
            query = f"{operator}({metric}{rule}) by ({group})"
        else:
            query = f"{metric}{rule}"

        return query
