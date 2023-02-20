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
Description: The calibrator which normalizes the anomaly score to the normal distribution
"""

from typing import List

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import fmin
from scipy.stats import norm

from anteater.model.algorithms.base import Serializer
from anteater.model.algorithms.n_sigma import n_sigma
from anteater.utils.common import divide
from anteater.utils.log import logger


class Calibrator(Serializer):
    """The base calibrator for mapping the values to normal distribution

    - implement calibrator referring to:
      https://github.com/salesforce/Merlion/blob/main/merlion/post_process/calibrate.py
    """

    filename = "calibrator.json"

    def __init__(self, max_score: float = 1000, anchors=None, **kwargs):
        """The calibrator initializer"""
        self.max_score = max_score
        self.anchors = anchors
        self._interpolator = None

        self.update_interpolator(self.anchors)

    def __call__(self, values):
        """The callable object"""
        if self._interpolator is None:
            return values

        b = self.anchors[-1][0]
        m = self._interpolator.derivative()(self.anchors[-1][0])

        y = np.maximum(self._interpolator(np.abs(values)), 0) * np.sign(values)
        idx = np.abs(values) > b
        if idx.any():
            sub = values[idx]
            y[idx] = np.sign(sub) * \
                ((np.abs(sub) - b) * m + self._interpolator(b))

        return y

    def update_interpolator(self, anchors):
        """Updates interpolator and anchors"""
        if anchors is None or len(anchors) < 2:
            self.anchors = None
            self._interpolator = None
        else:
            self.anchors = anchors
            self._interpolator = PchipInterpolator(*zip(*anchors))

    def fit_transform(self, values: List[float], retrain=False):
        """Train the calibration parameters"""
        if self._interpolator is not None and not retrain:
            return self(values)

        values = np.abs(values)

        targets = [0, 0, 0.5, 1, 1.5, 2]
        inputs = np.quantile(values, 2 * norm.cdf(targets) - 1).tolist()

        ub = np.sqrt(2 * np.log(len(values)))
        x_max = values.max()
        if self.max_score < x_max:
            logger.warning(
                f'Updating self.max_score from {self.max_score:.2f} '
                f'to {x_max * 2:.2f}.')
            self.max_score = x_max * 2
        if ub > 4:
            targets.append(ub)
            inputs.append(values.max())
            targets.append(ub + 1)
            inputs.append(min(self.max_score, 2 * x_max))
        else:
            targets.append(5)
            inputs.append(min(self.max_score, 2 * x_max))

        targets = np.asarray(targets)
        inputs = np.asarray(inputs)
        valid = np.concatenate(
            ([True], np.abs(inputs[1:] - inputs[:-1]) > 1e-8))
        self.update_interpolator(list(zip(inputs[valid], targets[valid])))

        return self(values)


class ErrorCalibrator(Serializer):
    """Predict- or reconstruction- errors calibrator which smooths
    the errors those are below a static or dynamic threshold
    """

    filename = "error_calibrator.json"

    def __init__(self, fixed_threshold=False, **kwargs):
        """The error calibrator initializer"""
        self.fixed_threshold = fixed_threshold
        self.n = kwargs.get('n', 3)
        self.quantile = kwargs.get('quantile', 0.99)
        self.min_threshold = kwargs.get('min_threshold', 0)
        self.alpha = kwargs.get('alpha', 0.02)

        self._z_range = (2, 10)
        self._min_z = 1.5

    def __call__(self, errors, *args, **kwargs):
        """The callable object smooths the errors below a threshold"""
        threshold = self._get_threshold(errors)
        below_error = np.max(errors[errors < threshold])

        return np.where(errors < threshold, 0, errors)

    def fit_transform(self, scores, *args, **kwargs):
        """Fit and transform data source"""
        if self.quantile is not None and self.quantile > 0:
            self.min_threshold = np.quantile(scores, self.quantile)

    def _get_threshold(self, errors):
        """Gets the threshold based on the errors"""
        if self.fixed_threshold:
            threshold = self._static_threshold(errors)
        else:
            threshold = self._dynamic_threshold(errors)

        return max(threshold, self.min_threshold)

    def _static_threshold(self, errors):
        """Computes the static threshold by n-sigma method"""
        _, mean, std = n_sigma(
            errors, obs_size=25, n=self.n, method='max')

        return mean + self.n * std

    def _dynamic_threshold(self, errors):
        """Computes the dynamic threshold based on unsupervised method"""
        mean = np.mean(errors)
        std = np.std(errors)

        best_z = self._z_range[0]
        best_score = np.inf
        for z in range(*self._z_range):
            best = fmin(self._z_score, z, args=(errors, mean, std), full_output=True, disp=False)
            z, cost = best[0:2]
            if cost < best_score:
                best_score = cost
                best_z = z[0]

        best_z = max(self._min_z, best_z)
        return mean + best_z * std

    def _z_score(self, z, errors, mean, std):
        """Cost function computes a z score which represents
        how best of current split errors
        """
        threshold = mean + z * std

        delta_mean, delta_std = self._deltas(errors, threshold, mean, std)
        above, consecutive = self._num_above_sequence(errors, threshold)

        numerator = -(divide(delta_mean, mean) + divide(delta_std, std))
        denominator = above  # + consecutive ** 2

        if denominator == 0:
            return np.inf

        return numerator + self.alpha * denominator

    def _deltas(self, errors, threshold, mean, std):
        """Computes the delta values between mean and below mean, std and below std"""
        below = errors[errors <= threshold]
        if not below.any():
            return 0, 0

        return mean - below.mean(), std - below.std()

    def _num_above_sequence(self, errors, threshold):
        """Computes the number of sequence above the threshold"""
        above = errors > threshold
        total_above = len(errors[above])

        above = pd.Series(above)
        shift = above.shift(1)
        change = above != shift

        total_consecutive = sum(above & change)

        return total_above, total_consecutive
