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
Description: The threshold class for dynamic anomaly score thresholds selection
"""

from collections import deque

import numpy as np

from anteater.model.algorithms.base import Serializer
from anteater.utils.common import divide


class Threshold(Serializer):
    """The threshold post process"""

    filename = "threshold.json"

    def __init__(self, alm_threshold: int = 2, abs_score: bool = True, quantile: float = 0.99, **kwargs):
        """The threshold post process initializer"""
        self.alm_threshold = alm_threshold
        self.abs_score = abs_score
        self.quantile = quantile

        self._history_scores = deque(maxlen=100000)
        self._latest_alm_threshold = 0
        self._p = 1.2

    def __call__(self, scores):
        """The callable object"""
        self._history_scores.extend(scores[-12:])

        threshold = max(self._latest_alm_threshold, self.alm_threshold)

        updated_scores = np.where(scores >= threshold, scores, 0.0)

        self.update_alm_th(scores)

        return updated_scores

    def update_alm_th(self, scores):
        """Calculates new threshold for alarms"""
        threshold = max(self._latest_alm_threshold, self.alm_threshold)
        self._latest_alm_threshold = (np.average(scores) + threshold) / 2

    def fit(self, scores):
        """Train the threshold of post process"""
        if self.abs_score:
            scores = np.abs(scores)

        if self.quantile is not None and self.quantile > 0:
            self.alm_threshold = np.quantile(scores, self.quantile)

        self._latest_alm_threshold = 0

    def error_rate(self, look_back):
        """Computes the error rate"""
        hist_scores = np.array(self._history_scores)[-look_back:]

        alarms = hist_scores[np.where(hist_scores >= self.alm_threshold)]

        if alarms and hist_scores:
            return divide(len(alarms), len(hist_scores))

        return 0


class DynamicThreshold(Serializer):
    """The dynamic threshold post process

    Compare the adjacent scores increasing rate with min_percent, then
    pruning the scores by removing those have a lower increasing rate.
    """

    filename = "dy_threshold.json"

    def __init__(self, min_percent=-1, **kwargs):
        """The dynamic threshold post process initializer"""
        self.min_percent = min_percent

    def __call__(self, scores):
        """The callable object"""
        np.seterr(all='ignore')
        size = len(scores)
        left = scores[:size - 1]
        right = scores[-(size - 1):]
        rate = np.divide(right - left, left)
        rate = np.nan_to_num(rate)
        indices = np.argwhere(rate > self.min_percent) + 1

        pruned_scores = np.zeros(size)
        for _idx in indices.flatten():
            pruned_scores[_idx] = scores[_idx]

        return pruned_scores

    def fit(self, *args, **kwargs):
        """Train the threshold of post process"""
        pass
