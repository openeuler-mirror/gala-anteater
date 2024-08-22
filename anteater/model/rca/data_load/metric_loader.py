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

from typing import List, Union

from anteater.model.rca.data_load.utils import datetime
from anteater.model.rca.data_load.time_series import TimeSeries
from anteater.model.rca.data_load.utils.timer import timer
from anteater.utils.log import logger


def get_query(metric: str,
              label_name: Union[str, List] = None, label_value: Union[str, List] = None,
              operator_name: str = None, operator_value: float = None):
    """Gets aggregated query patterns

        Such as:
            - 1. gala_gopher_bind_sends{machine_id="1234"}
            - 2. sum(gala_gopher_bind_sends) by (machine_id)
            - 3. sum(gala_gopher_bind_sends{machine_id="1234"}) by (machine_id)
            - 4. quantile(0.7, gala_gopher_bind_sends{machine_id="1234"}) by (machine_id)
    """
    rule = ""
    if label_value:
        if (isinstance(label_name, str) and isinstance(label_value, list)) or (
                isinstance(label_name, list) and isinstance(label_value, str)):
            raise ValueError(f"The label_name and label_value are of different types,"
                             f"type(label_name): {type(label_name)},"
                             f"type(label_value): {type(label_value)}.")
        if isinstance(label_value, list):
            pairs = ",".join([f"{n}=~'{v}'" for n, v in zip(label_name, label_value)])
            rule = f"{{{pairs}}}"
        elif isinstance(label_value, str):
            rule = f"{{{label_name}=~'{label_value}'}}"

    group = ""
    if isinstance(label_name, list):
        group = ",".join([k for k in label_name])
    elif isinstance(label_name, str):
        group = label_name

    if operator_name and operator_value:
        query = f"{operator_name}({operator_value}, {metric}{rule}) by ({group})"
    elif operator_name:
        query = f"{operator_name}({metric}{rule}) by ({group})"
    else:
        query = f"{metric}{rule}"

    return query


def get_query2(
        metric: str, operator: str = None, value: float = None, keys: Union[str, List] = None, **labels):
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
        pairs = ",".join([f"{n}='{v}'" for n, v in labels.items()])
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


def get_kafka_query(
        metric: str, operator: str = None, value: float = None, keys: Union[str, List] = None, **labels):
    if operator and not keys:
        raise ValueError("Please provide param 'keys' when specified 'operator'!")

    rule = ""
    if labels:
        pairs = ",".join([f"{n}='{v}'" for n, v in labels.items()])
        rule = f"{{{pairs}}}"

    group = ""
    if isinstance(keys, list):
        group = ",".join([k for k in keys])
    elif isinstance(keys, str):
        group = keys

    if operator and value:
        query = f"{operator}({value}, {metric}{rule}[1m])"
    elif operator:
        query = f"{operator}({metric}{rule}[1m])"
    else:
        query = f"{metric}{rule}"

    return query


def get_tomcat_query(
        metric: str, operator: str = None, value: float = None, keys: Union[str, List] = None, **labels):
    if operator and not keys:
        raise ValueError("Please provide param 'keys' when specified 'operator'!")

    rule = ""
    if labels:
        pairs = ",".join([f"{n}='{v}'" for n, v in labels.items()])
        rule = f"{{{pairs}}}"

    group = ""
    if isinstance(keys, list):
        group = ",".join([k for k in keys])
    elif isinstance(keys, str):
        group = keys

    if operator and value:
        query = f"avg({operator}({value}, {metric}{rule}[1m]))"
    elif operator:
        query = f"avg({operator}({metric}{rule}[1m]))"
    else:
        query = f"avg({metric}{rule})"

    return query


class MetricLoader:
    """
    The metric loader that consumes raw data from PrometheusAdapter,
    then convert them to dataframe
    """

    def __init__(self, client) -> None:
        """The Metrics Loader initializer"""
        self.provider = client

    def get_metric(self, start: datetime, end: datetime, metric: str, **kwargs) -> List[TimeSeries]:
        """Get target metric time series data

        :kwargs
            - label_name: Union[str, List] = None,
            - label_value: Union[str, List] = None,
            - operator_name: str = None,
            - operator_value: float = None)

        :return List of TimeSeries
        """
        query = get_query2(metric, **kwargs)
        time_series = self.provider.range_query(start, end, metric, query, is_single=False)

        return time_series

    def get_kafka_metric(self, start: datetime, end: datetime, metric: str, **kwargs) -> List[TimeSeries]:
        # query = "rate(kafka_topic_partition_current_offset{job='vm10-10.109.56.225.kafka'}[1m])"
        query = get_kafka_query(metric, **kwargs)
        time_series = self.provider.range_query(start, end, metric, query, is_single=False)

        return time_series

    def get_tomcat_metric(self, start: datetime, end: datetime, metric: str, **kwargs) -> List[TimeSeries]:
        # query = "avg(rate(servlet_request_seconds_bucket{job="vm08-10.109.56.226.tomcat"}[1m]))"
        query = get_tomcat_query(metric, **kwargs)
        time_series = self.provider.range_query(start, end, metric, query, is_single=False)

        return time_series

    def get_single_metric(self, start: datetime, end: datetime, metric: str, **kwargs) -> List[TimeSeries]:
        query = get_query(metric, **kwargs)

        time_series = self.provider.range_query(start, end, metric, query, is_single=True)

        return time_series

    @timer
    def get_unique_machines(self, start: datetime, end: datetime, metrics: List[str]) -> List[str]:
        """Gets the unique machine ids based on the metrics"""
        machine_ids = self.get_unique_label(start, end, metrics, label_name="machine_id")
        if not machine_ids:
            logger.logger.warning(f'Empty unique machine ids on given metrics!')
            logger.logger.debug(f'metrics: {";".join(metrics)}')
        else:
            logger.logger.debug(f'Got {len(machine_ids)} unique machine_ids on given metrics')
            logger.logger.debug(f'machine ids: {";".join(machine_ids)}')
            logger.logger.debug(f'metrics: {";".join(metrics)}')
        return machine_ids

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
        if self.provider.step >= 0:
            return max((end - start) // self.provider.step, 1)
        else:
            return 0
