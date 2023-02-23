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

from itertools import groupby
from typing import List

from anteater.core.anomaly import Anomaly
from anteater.core.kpi import KPI
from anteater.core.time_series import TimeSeriesScore
from anteater.model.detector.base import Detector
from anteater.model.algorithms.smooth import smoothing
from anteater.model.algorithms.n_sigma import n_sigma
from anteater.source.metric_loader import MetricLoader
from anteater.utils.common import divide
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger


class NSigmaDetector(Detector):
    """The n-sigma anomaly detector"""

    def __init__(self, data_loader: MetricLoader):
        """The detector base class initializer"""
        super().__init__(data_loader)

    def detect_kpis(self, kpis: List[KPI]):
        """Executes anomaly detection on kpis"""
        start, end = dt.last(minutes=1)
        machine_ids = self.get_unique_machine_id(start, end, kpis)
        anomalies = []
        for _id in machine_ids:
            for kpi in kpis:
                anomalies.extend(self.detect_signal_kpi(kpi, _id))

        return anomalies

    def detect_signal_kpi(self, kpi, machine_id: str) -> List[Anomaly]:
        """Detects kpi based on signal time series anomaly detection model"""
        outlier_ratio_th = kpi.params['outlier_ratio_th']
        ts_scores = self.calculate_n_sigma_score(
            kpi.metric, kpi.description, machine_id, **kpi.params)
        if not ts_scores:
            logger.warning(f'Key metric {kpi.metric} is null on the target machine {machine_id}!')
            return []

        ts_scores = [t for t in ts_scores if t.score >= outlier_ratio_th]
        anomalies = [
            Anomaly(
                machine_id=machine_id,
                metric=_ts_score.ts.metric,
                labels=_ts_score.ts.labels,
                score=_ts_score.score,
                entity_name=kpi.entity_name,
                description=kpi.description)
            for _ts_score in ts_scores
        ]

        return anomalies

    def calculate_n_sigma_score(self, metric, description, machine_id: str, **kwargs)\
            -> List[TimeSeriesScore]:
        """Calculate kpi anomaly scores based on three sigma scores"""
        method = kwargs.get('method', 'abs')
        look_back = kwargs.get('look_back')
        smooth_params = kwargs.get('smooth_params')
        obs_size = kwargs.get('obs_size')
        n = kwargs.get('n', 3)
        start, end = dt.last(minutes=look_back)
        point_count = self.data_loader.expected_point_length(start, end)
        ts_list = self.data_loader.get_metric(start, end, metric, machine_id=machine_id)
        ts_scores = []
        for _ts in ts_list:
            dedup_values = [k for k, g in groupby(_ts.values)]
            if sum(_ts.values) == 0 or \
               len(_ts.values) < point_count * 0.6 or \
               len(_ts.values) > point_count * 1.5 or \
               all(x == _ts.values[0] for x in _ts.values):
                ratio = 0
            elif len(dedup_values) < point_count * 0.6:
                ratio = 0
            else:
                smoothed_val = smoothing(_ts.values, **smooth_params)
                outlier, mean, std = n_sigma(
                    smoothed_val, obs_size=obs_size, n=n, method=method)
                ratio = divide(len(outlier), obs_size)

            ts_scores.append(TimeSeriesScore(ts=_ts, score=ratio, description=description))

        return ts_scores
