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


from typing import Dict

import numpy as np
from pandas import DataFrame

from anteater.model.algorithms.compress_sense import CSAnomalyDetector


def standard_norm(data):
    _range = np.max(data) - np.min(data)
    if _range == 0:
        return np.zeros_like(data) + 0.5

    return np.divide(data - np.min(data), _range)


class JumpStarter:
    """The noise preprocessor on training dataset based on JumpStarter model

    It refers the model implementation here:
        - https://github.com/NetManAIOps/JumpStarter
    """

    def __init__(self, config: Dict):
        """The JumpStarter preprocessor initializer"""
        self.config = config

        self.random_state = self.config.get('random_state')
        self.lesinn_t = self.config.get('lesinn_t')
        self.lesinn_phi = self.config.get('lesinn_phi')
        self.moving_average_window = self.config.get('moving_average_window')
        self.moving_average_stride = self.config.get('moving_average_stride')
        self.anomaly_score_example_percentage = self.config.get('anomaly_score_example_percentage')
        self.anomaly_distance_topn = self.config.get('anomaly_distance_topn')

        self.rec_window = self.config.get('rec_window')
        self.rec_stride = self.config.get('rec_stride')
        self.det_window = self.config.get('det_window')
        self.det_stride = self.config.get('det_stride')
        self.rec_windows_per_cycle = self.config.get('rec_windows_per_cycle')
        self.save_path = self.config.get('save_path')

        self.workers = self.config.get('workers')
        self.anomaly_scoring = self.config.get('anomaly_scoring')
        self.sample_score_method = self.config.get('sample_score_method')
        self.cluster_threshold = self.config.get('cluster_threshold')
        self.sample_rate = self.config.get('sample_rate')
        self.latest_windows = self.config.get('latest_windows')
        self.scale = self.config.get('scale')
        self.rho = self.config.get('rho')
        self.sigma = self.config.get('sigma')
        self.retry_limit = self.config.get('retry_limit')
        self.without_grouping = self.config.get('without_grouping')
        self.ratio = self.config.get('ratio')

    def transform(self, x: DataFrame):
        """"""
        n, d = x.shape
        if n < self.rec_window * self.rec_windows_per_cycle:
            raise ValueError('data point count less than 1 cycle')

        data = x.values
        for i in range(d):
            data[:, i] = standard_norm(data[:, i])

        detector = CSAnomalyDetector(
            workers=self.workers,
            cluster_threshold=self.cluster_threshold,
            sample_rate=self.sample_rate,
            sample_score_method=self.sample_score_method,
            distance=self.anomaly_scoring,
            scale=self.scale,
            rho=self.rho,
            sigma=self.sigma,
            random_state=self.random_state,
            retry_limit=self.retry_limit,
            without_grouping=self.without_grouping)

        rec, _ = detector.reconstruct(
            data, self.rec_window, self.rec_windows_per_cycle, self.rec_stride)

        score = detector.predict(data, rec, self.det_window, self.det_stride)
        std = np.std(score)
        mean = np.mean(score)
        threshold = mean + self.ratio * std
        pred_pos = np.array(score > threshold, dtype=np.int16)

        return x.iloc[:, pred_pos]
