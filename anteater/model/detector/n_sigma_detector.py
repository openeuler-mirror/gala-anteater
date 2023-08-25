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
from typing import List, Tuple

from anteater.core.anomaly import Anomaly
from anteater.core.kpi import KPI
from anteater.core.ts import TimeSeries
from anteater.model.detector.base import Detector
from anteater.model.algorithms.smooth import smoothing
from anteater.model.algorithms.n_sigma import n_sigma
from anteater.utils.common import divide
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger


class NSigmaDetector(Detector):
    """The n-sigma anomaly detector"""

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
        ts_scores = self.cal_n_sigma_score(
            kpi.metric, machine_id, **kpi.params)
        if not ts_scores:
            logger.warning('Key metric %s is null on the target machine %s!',
                           kpi.metric, machine_id)
            return []

        ts_scores = [t for t in ts_scores if t[1] >= outlier_ratio_th]
        anomalies = [
            Anomaly(
                machine_id=machine_id,
                metric=_ts.metric,
                labels=_ts.labels,
                score=_score,
                entity_name=kpi.entity_name)
            for _ts, _score in ts_scores
        ]

        return anomalies

    def cal_n_sigma_score(self, metric, machine_id: str, **kwargs) \
            -> List[Tuple[TimeSeries, int]]:
        """Calculates metrics' ab score based on n-sigma method"""
        method = kwargs.get('method', 'abs')
        look_back = kwargs.get('look_back')
        smooth_params = kwargs.get('smooth_params')
        obs_size = kwargs.get('obs_size')
        n = kwargs.get('n', 3)
        start, end = dt.last(minutes=look_back)
        point_count = self.data_loader.expected_point_length(start, end)
        ts_list = self.data_loader.get_metric(
            start, end, metric, machine_id=machine_id)
        ts_scores = []
        for _ts in ts_list:
            dedup_values = [k for k, g in groupby(_ts.values)]
            if sum(_ts.values) == 0 or \
               len(_ts.values) < point_count * 0.6 or \
               len(_ts.values) > point_count * 1.5 or \
               all(x == _ts.values[0] for x in _ts.values):
                score = 0
            elif len(dedup_values) < point_count * 0.6:
                score = 0
            else:
                smoothed_val = smoothing(_ts.values, **smooth_params)
                outlier, _, _ = n_sigma(
                    smoothed_val, obs_size=obs_size, n=n, method=method)
                score = divide(len(outlier), obs_size)

            ts_scores.append((_ts, score))

        return ts_scores
