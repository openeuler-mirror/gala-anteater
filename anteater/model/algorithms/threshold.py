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

import copy
import json
import os
import stat
from collections import deque

import numpy as np

from anteater.utils.common import divide
from anteater.utils.log import logger


class Threshold:
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
            logger.warning("Unknown model file, load default threshold model!")
            return Threshold(**kwargs)

        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(config_file, os.O_RDONLY, modes), "r") as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

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
