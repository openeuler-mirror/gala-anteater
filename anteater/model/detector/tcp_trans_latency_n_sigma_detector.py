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

from typing import List

from anteater.core.time_series import TimeSeriesScore
from anteater.model.detector.n_sigma_detector import NSigmaDetector
from anteater.source.metric_loader import MetricLoader
from anteater.utils.datetime import DateTimeManager as dt


class TcpTransLatencyNSigmaDetector(NSigmaDetector):
    """The three sigma anomaly detector"""

    def __init__(self, data_loader: MetricLoader, method: str):
        """The detector base class initializer"""
        super().__init__(data_loader, method)

    def cal_anomaly_score(self, metric, description, machine_id: str) \
            -> List[TimeSeriesScore]:
        """Calculate metric anomaly scores based on max values"""
        start, end = dt.last(minutes=2)
        point_count = self.data_loader.expected_point_length(start, end)
        ts_scores = []
        ts_list = self.data_loader. \
            get_metric(start, end, metric, label_name='machine_id', label_value=machine_id)
        for _ts in ts_list:
            if sum(_ts.values) == 0 or \
                    len(_ts.values) < point_count * 0.5 or \
                    len(_ts.values) > point_count * 1.5:
                score = 0
            else:
                score = max(_ts.values)

            ts_scores.append(TimeSeriesScore(ts=_ts, score=score, description=description))

        return ts_scores
