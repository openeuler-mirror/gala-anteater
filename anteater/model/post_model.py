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
"""
Time:
Author:
Description: The post process which aims to find the top n anomalies.
"""

import math
from typing import List, Tuple, Any

from anteater.config import AnteaterConf
from anteater.model.algorithms.spectral_residual import SpectralResidual
from anteater.source.metric_loader import MetricLoader
from anteater.utils.data_load import load_metric_operator
from anteater.utils.datetime import datetime_manager
from anteater.utils.time_series import TimeSeries


class PostModel:
    """The post model which aims to recommend some key metrics for abnormal events"""

    def __init__(self, config: AnteaterConf) -> None:
        """The post model initializer"""
        self.config = config
        self.metric_operators = load_metric_operator()
        self.unique_metrics = set([m for m, _ in self.metric_operators])
        self.data_loader = MetricLoader(self.config)

    @staticmethod
    def predict(sr_model: SpectralResidual, values: List) -> float:
        """Predicts anomalous score for the time series values"""
        if all(x == values[0] for x in values):
            return -math.inf

        scores = sr_model.compute_score(values)

        return max(scores[-12: -1])

    def get_all_metric(self, start, end, machine_id: str) -> List[TimeSeries]:
        """Gets all metric labels and values"""
        time_series_list = []
        for metric in self.unique_metrics:
            time_series = self.data_loader.get_metric(start, end, metric,
                                                      label_name="machine_id", label_value=machine_id)
            time_series_list.extend(time_series)

        return time_series_list

    def top_n_anomalies(self, machine_id: str, top_n: int) -> List[Tuple[TimeSeries, Any]]:
        """Finds top n anomalies during a period for the target machine"""
        start, end = datetime_manager.last(minutes=6)
        time_series_list = self.get_all_metric(start, end, machine_id)
        point_count = self.data_loader.expected_point_length(start, end)
        sr_model = SpectralResidual(12, 24, 50)

        result = []
        for time_series in time_series_list:
            if len(time_series.values) < point_count * 0.9 or\
               len(time_series.values) > point_count * 1.5:
                continue

            score = self.predict(sr_model, time_series.values)

            if math.isnan(score) or math.isinf(score):
                continue

            result.append((time_series, score))

        result = sorted(result, key=lambda x: x[1], reverse=True)

        return result[0: top_n]
