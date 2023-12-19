#!/usr/bin/python3
# ******************************************************************************
# Copyright (c) 2023 Huawei Technologies Co., Ltd.
# gala-anteater is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# ******************************************************************************/


from enum import Enum
import json
from os import path

from anteater.utils.constants import ANTEATER_CONFIG_PATH
from anteater.utils.log import logger


class MetricType(Enum):
    """The metric type"""
    GAUGE = 0
    HISTOGRAM = 1
    COUNTER = 2
    SUMMARY = 3

    @staticmethod
    def from_str(label: str):
        """Trans str to Enum type"""
        if label.lower() == 'gauge':
            return MetricType.GAUGE
        elif label.lower() == 'histogram':
            return MetricType.HISTOGRAM
        elif label.lower() == 'counter':
            return MetricType.COUNTER
        elif label.lower() == 'summary':
            return MetricType.SUMMARY
        else:
            raise ValueError(f'Unknown metric type: {label}')


class MetricAggregation(Enum):
    """The metric aggregation"""
    AVG = 0
    MAX = 1
    MIN = 2
    SUM = 3

    @staticmethod
    def from_str(label: str):
        """Trans str to Enum aggregation"""
        if label.lower() == 'avg':
            return MetricAggregation.AVG
        elif label.lower() == 'max':
            return MetricAggregation.MAX
        elif label.lower() == 'min':
            return MetricAggregation.MIN
        elif label.lower() == 'sum':
            return MetricAggregation.SUM
        else:
            raise ValueError(f'Unknown metric Aggregation: {label}')


class MetricInfo:
    """The metric info class

    Which contains metric info, such as en- or zh- language
    description, metric type, etc
    """

    file_name = "metricinfo.json"

    def __init__(self) -> None:
        """The metric info class initializer"""
        self.metric2type = {}
        self.metric2en = {}
        self.metric2zh = {}
        self.metric2aggregation = {}
        self.metric2classification = {}

        self._load_desc()

    def get_type(self, metric) -> MetricType:
        """Gets metric type base on the metric name"""
        return self.metric2type.get(metric, None)

    def get_en(self, metric: str) -> str:
        """Gets english description"""
        return self.metric2en.get(metric, '')

    def get_zh(self, metric: str) -> str:
        """Gets chinese description"""
        return self.metric2zh.get(metric, '')

    def get_aggregation(self, metric) -> str:
        """Gets metric aggregation base on the metric name"""
        return self.metric2aggregation.get(metric, None)

    def get_classification(self, metric) -> str:
        """Gets metric classification base on the metric name"""
        return self.metric2classification.get(metric, None)

    def append_type(self, metric: str, typ: MetricType) -> None:
        """Appends metric and type pair"""
        self.metric2type[metric] = typ

    def append_en(self, metric: str, des: str) -> None:
        """Appends metric and en- desc pair"""
        self.metric2en[metric] = des

    def append_zh(self, metric: str, des: str) -> None:
        """Appends metric and zh- desc pair"""
        self.metric2zh[metric] = des

    def append_aggregation(self, metric: str, aggregation: MetricAggregation) -> None:
        """Appends metric and aggregation pair"""
        self.metric2aggregation[metric] = aggregation

    def append_classification(self, metric: str, classification: str) -> None:
        """Appends metric and classification pair"""
        self.metric2classification[metric] = classification

    def in_type(self, metric: str) -> bool:
        """Whether metric is already in metric2type or not"""
        return metric in self.metric2type

    def in_en(self, metric: str) -> bool:
        """Whether metric is already in en desc or not"""
        return metric in self.metric2en

    def in_zh(self, metric: str) -> bool:
        """Whether metric is already in zh desc or not"""
        return metric in self.metric2zh

    def in_aggregation(self, metric: str) -> bool:
        """Whether metric is already in metric2aggregation or not"""
        return metric in self.metric2aggregation

    def in_classification(self, metric: str) -> bool:
        """Whether metric is already in metric2classification or not"""
        return metric in self.metric2classification

    def _load_desc(self):
        """Loads metric info from the file"""
        folder_path = path.realpath(ANTEATER_CONFIG_PATH)
        abs_path = path.join(folder_path, self.file_name)

        with open(abs_path, 'r', encoding='utf-8') as f_out:
            try:
                items = json.load(f_out)
            except json.JSONDecodeError as e:
                logger.error('JSONDecodeError: when parsing file %s',
                             path.basename(abs_path))
                raise e

        for item in items:
            metric = item.get('metric', '')
            if not metric:
                raise KeyError('Empty metric name in config file '
                               f'{path.basename(abs_path)}')

            if self.in_type(metric) or self.in_en(metric) or self.in_zh(metric):
                raise KeyError(f'Duplicated metric \'{metric}\' in config '
                               f'file {path.basename(abs_path)}')
            if self.in_aggregation(metric) or self.in_classification(metric):
                raise KeyError(f'Duplicated metric \'{metric}\' in config '
                               f'file {path.basename(abs_path)}')

            typ = item.get('data_type', '')
            if not typ:
                raise KeyError(f'Empty metric type on \'{metric}\'')
            typ = MetricType.from_str(typ)
            self.append_type(metric, typ)

            desc_en = item.get('en', '')
            desc_zh = item.get('zh', '')
            if not desc_en and not desc_zh:
                raise KeyError(f'Empty en and zh desc on \'{metric}\'')
            if desc_en:
                self.append_en(metric, desc_en)
            if desc_zh:
                self.append_zh(metric, desc_zh)

            aggregation = item.get('aggregation', '')
            if not aggregation:
                raise KeyError(f'Empty metric aggregation on \'{metric}\'')
            aggregation = MetricAggregation.from_str(aggregation)
            self.append_aggregation(metric, aggregation)

            classification = item.get('data_classification', '')
            if not classification:
                raise KeyError(f'Empty metric classification on \'{metric}\'')
            self.append_classification(metric, classification)
