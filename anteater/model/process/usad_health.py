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

import numpy as np


def sigmoid(z, alpha, offset=0.5):
    """Sigmoid activation functions"""
    y = 1 / (1 + np.exp(-alpha * (z - offset)))
    return y


def relu(x, alpha, offset, liner_criteria=None):
    """ReLU activation functions"""
    if liner_criteria is None:
        liner_criteria = 0.5
    if x > liner_criteria:
        return sigmoid(x, alpha=alpha, offset=offset)
    elif x <= 0:
        return 0
    else:
        return x


class HealthMulti:
    def __init__(self, train_std, criterion_np=None):
        """Multi-metrics health detection"""

        self.train_std = train_std
        self.criterion_np = criterion_np
        self.window_length_limit = 5
        self.anomaly_window = {}
        self.sustained_healthy_number = 0
        self.anomaly_happen = False

        self.dev_weight = 0.5
        self.dur_weight = 0.5

        self.update_flag = False
        self.alarms = []
        self.health_info = {}

    @property
    def health_score_info(self):
        res = []
        for v in self.health_info.values():
            res.append(v)
        return res

    def get_weight(self, x_np, error, threshold):
        """Gets dev_weight and dur_weight"""
        if self.criterion_np is not None:
            if len(np.where(x_np >= self.criterion_np)[0]) != 0:
                self.dev_weight = 1.0
                self.dur_weight = 0.0
            else:
                self.dev_weight = 0.0
                self.dur_weight = 1.0
        else:
            self.dev_weight = 0.0
            self.dur_weight = 1.0

    def get_deviation_score(self, x_np, error, threshold):
        """Gets deviation score"""
        if error >= threshold:
            ratio = 1.0
            if self.criterion_np is not None:
                if len(np.where(x_np >= self.criterion_np)[0]) != 0:
                    return 1.0
                else:
                    ratio = np.max(np.abs(x_np - threshold) /
                                   (self.criterion_np - threshold + 1e-4))
            ratio = min(ratio, 1.0)
            score = (error - threshold) / self.train_std
            score = relu(ratio, 14, 0.5) * sigmoid(score, 5 / 6)
        else:
            score = 0
        return score

    def get_sustained_score(self, index, error, threshold):
        """Gets sustained score"""
        if error > threshold:
            self.anomaly_window[index] = 1
            self.sustained_healthy_number = 0
            if len(self.anomaly_window) <= self.window_length_limit:
                if np.sum([v for k, v in self.anomaly_window.items()]) < 3:
                    score = 0
                else:
                    score = 1
                    self.anomaly_happen = True
            else:
                score = 1
                self.anomaly_happen = True

        else:
            self.sustained_healthy_number = self.sustained_healthy_number + 1
            if self.anomaly_happen:
                if self.sustained_healthy_number < 3:
                    score = 1
                    self.anomaly_window[index] = 0
                else:
                    score = 0
                    self.anomaly_window = {}
                    self.anomaly_happen = False
                    self.update_flag = False
            else:
                score = 0
                if self.sustained_healthy_number < 3:
                    self.anomaly_window[index] = 0
                else:
                    self.anomaly_window = {}
                    self.update_flag = False
                    self.anomaly_happen = False

        return score

    def fit(self, index, x_np, error, threshold):
        """Gets health score"""
        deviation_score = self.get_deviation_score(x_np, error, threshold)
        sustained_score = self.get_sustained_score(index, error, threshold)
        self.get_weight(x_np, error, threshold)
        health_score = 1 - (deviation_score * self.dev_weight +
                            sustained_score * self.dur_weight)
        self.health_info[index] = [
            health_score, deviation_score, sustained_score, self.dur_weight
        ]
        self.update(health_score, sustained_score)

    def predict(self, threshold):
        """Gets anomaly alarms"""
        for index, score_l in self.health_info.items():
            if score_l[0] < threshold:
                self.alarms.append(index)
        return self.alarms

    def update(self, health_score, new_sustained_score):
        """Adjusts health score"""
        if health_score < 0.2 and not self.update_flag:
            for _id in self.anomaly_window.keys():
                origin_health_score_info = self.health_info.get(_id, [])
                if len(origin_health_score_info) == 0:
                    self.update_flag = False
                    return
                new_health_score = 1 - \
                    (origin_health_score_info[1] * self.dev_weight +
                     new_sustained_score * self.dur_weight)
                origin_health_score_info[0] = new_health_score
                origin_health_score_info[2] = new_sustained_score
                self.health_info[_id] = origin_health_score_info
            self.update_flag = True
        else:
            pass
