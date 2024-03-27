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
Description: The base class for time series data fetching
"""

from abc import abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List

import requests

from anteater.utils.log import logger
from anteater.core.ts import TimeSeries


class TimeSeriesProvider:
    """The base class for time series data collector"""
    def __init__(self, url, step=5):
        """The PrometheusAdapter client initializer"""
        self.url = url
        self.step = step

    @staticmethod
    def chunks(start_time, end_time, hours=6):
        """Split a duration (from start time to end time) to multi-disjoint intervals"""
        if start_time >= end_time:
            raise ValueError("The start_time greater or equal than end_time!")

        if not isinstance(start_time, datetime):
            raise ValueError("The type of start_time isn't datetime!")

        if not isinstance(end_time, datetime):
            raise ValueError("The type of end_time isn't datetime!")

        _start = start_time
        _end = _start
        while _end < end_time:
            _end = min(_start + timedelta(hours=hours), end_time)
            yield _start, _end
            _start = _end

    @staticmethod
    def symbol_replace(metric, labels, is_single):
        if not is_single:
            return metric
        for k in labels:
            new_k = k.replace('/', '#')
            new_k = new_k.replace(':', '#')
            new_v = str(labels[k]).replace('/', '#')
            new_v = new_v.replace(':', '#')
            metric = f"{metric}@{new_k}={new_v}"
        return metric

    @abstractmethod
    def get_headers(self):
        """Gets the headers of requests"""

    def fetch(self, url, params: Dict, **args) -> List:
        """Fetches data from prometheus server by http request"""
        try:
            response = requests.get(url, params, timeout=30, **args)
        except requests.RequestException as e:
            logger.error(f"RequestException: {e}!")
            return []

        if response.status_code != 200:
            logger.error(f"{self.__class__.__name__}: "
                         f"requrests GET method response, "
                         f"status_code: {response.status_code}, "
                         f"reason: {response.reason}, "
                         f"url: {response.url}, ")
            return []

        response = response.json()
        result = []
        if response and response.get("status") == 'success':
            result = response.get('data', {}).get('result', [])
        else:
            logger.error(f"{self.__class__.__name__} get data failed, "
                         f"error: {response.get('error')}, "
                         f"query_url: {url}, "
                         f"params: {params}.")

        return result

    def range_query(self, start_time: datetime, end_time: datetime,
                    metric: str, query: str, is_single=False) -> List[TimeSeries]:
        """Range query time series data from PrometheusAdapter"""
        headers = self.get_headers()

        result = []
        tmp_index = {}
        for start, end in self.chunks(start_time, end_time):
            start, end = round(start.timestamp()), round(end.timestamp())
            params = {'query': query, 'start': start, 'end': end, 'step': self.step}

            data = self.fetch(self.url, params, headers=headers)

            for item in data:
                zipped_values = list(zip(*item.get('values')))
                time_stamps = list(zipped_values[0])
                values = [float(v) for v in zipped_values[1]]

                key = tuple(sorted(item.get('metric').items()))
                if key in tmp_index:
                    result[tmp_index.get(key)].extend(time_stamps, values)
                else:
                    labels = item.get('metric')
                    labels.pop('__name__', None)
                    metric = self.symbol_replace(metric, labels, is_single)
                    time_series = TimeSeries(
                        metric,
                        labels,
                        time_stamps,
                        values)
                    tmp_index[key] = len(result)
                    result.append(time_series)

        return result
