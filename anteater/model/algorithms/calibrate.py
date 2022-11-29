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

import copy
import json
import os
import stat
from typing import List

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm

from anteater.utils.log import logger


class Calibrator:
    """The calibrator which mapping the values to normal distribution"""

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

    @classmethod
    def from_dict(cls, config_dict):
        """Loads the object from the dict"""
        config_dict = copy.copy(config_dict)

        return cls(**config_dict)

    @classmethod
    def load(cls, folder: str, **kwargs):
        """Loads the model from the file"""
        config_file = os.path.join(folder, cls.filename)

        if not os.path.isfile(config_file):
            logger.warning("Unknown model file, load default calibrator model!")
            return Calibrator(**kwargs)

        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(config_file, os.O_RDONLY, modes), "r") as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

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

    def to_dict(self):
        """Dumps the object to the dict"""
        state_dict = {}
        for key, val in self.__dict__.items():
            if not key.startswith('_'):
                state_dict[key] = val

        return state_dict

    def save(self, folder):
        """Saves the model into the file"""
        config_dict = self.to_dict()
        modes = stat.S_IWUSR | stat.S_IRUSR
        config_file = os.path.join(folder, self.filename)
        with os.fdopen(os.open(config_file, os.O_WRONLY | os.O_CREAT, modes), "w") as f:
            f.truncate(0)
            json.dump(config_dict, f, indent=2)
