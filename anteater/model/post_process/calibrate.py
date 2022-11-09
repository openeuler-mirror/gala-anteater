# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
# licensed under the Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#     http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
# PURPOSE.
# See the Mulan PSL v2 for more details.
# ******************************************************************************/
"""
Time:
Author:
Description: The calibrator which normalizes the anomaly score to the normal distribution
"""

from typing import List

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm

from anteater.model.post_process.base import PostProcess
from anteater.utils.log import logger


class Calibrator(PostProcess):
    """The calibrator which mapping the values to normal distribution"""

    def __init__(self, max_score: float, anchors=None, **kwargs):
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

    def train(self, values: List[float], retrain=False):
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
