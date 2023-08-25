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

import numpy as np

from anteater.core.ts import TimeSeries
from anteater.model.algorithms.smooth import smoothing
from anteater.model.algorithms.n_sigma import n_sigma
from anteater.model.detector.n_sigma_detector import NSigmaDetector
from anteater.utils.common import divide
from anteater.utils.datetime import DateTimeManager as dt


class TcpTransLatencyNSigmaDetector(NSigmaDetector):
    """The three sigma anomaly detector"""

    def cal_n_sigma_score(self, metric, machine_id: str,
                          **kwargs) -> List[Tuple[TimeSeries, int]]:
        """Calculates anomaly scores based on n sigma scores"""
        method = kwargs.get('method', 'abs')
        look_back = kwargs.get('look_back')
        smooth_params = kwargs.get('smooth_params')
        obs_size = kwargs.get('obs_size')
        min_srtt = kwargs.get("min_srtt")
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
                if outlier and np.average(outlier) <= min_srtt:
                    score = 0
                else:
                    score = divide(len(outlier), obs_size)

            ts_scores.append((_ts, score))

        return ts_scores

    def cal_metric_ab_score(self, metric, machine_id: str) \
            -> List[Tuple[TimeSeries, float]]:
        """Calculate metric's anomaly scores based on max values"""
        start, end = dt.last(minutes=2)
        point_count = self.data_loader.expected_point_length(start, end)
        ts_scores = []
        ts_list = self.data_loader.get_metric(
            start, end, metric, machine_id=machine_id)
        for _ts in ts_list:
            if sum(_ts.values) == 0 or \
                    len(_ts.values) < point_count * 0.5 or \
                    len(_ts.values) > point_count * 1.5:
                score = 0
            else:
                score = max(_ts.values)

            ts_scores.append((_ts, score))

        return ts_scores
