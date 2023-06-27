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
import numpy as np


class ClipScaler:
    """Clip and scale fatures by a given range from the std

    Formula:
        1. x = min(max(mean - a * std, x)  mean + a * std)
        2. x = (x - mean) / std
    """
    def __init__(self, alpha=20) -> None:
        self.alpha = alpha

        self.mean = None
        self.std = None

    def fit(self, x):
        """Compute the mean and std to be used for later scaling"""
        if self.mean is None or self.std is None:
            self.mean = np.mean(x, axis=0)

            self.std = np.std(x, axis=0)
            for _v in self.std:
                if _v < 1e-4:
                    _v = 1

        return self

    def fit_transform(self, x):
        """Compute and scale the feature"""
        return self.fit(x).transform(x)

    def transform(self, x):
        """Scale the fatures by using computed mean and std"""
        if not self.mean or not self.std:
            raise ValueError(
                'Need to run fit() before transform data X.')

        for i in range(x.shape[0]):
            # compute clip value: (mean - a * std, mean + a * std)
            clip_value = self.mean + self.alpha * self.std
            temp = x[i] < clip_value
            x[i] = temp * x[i] + (1 - temp) * clip_value
            clip_value = self.mean - self.alpha * self.std
            temp = x[i] > clip_value
            x[i] = temp * x[i] + (1 - temp) * clip_value
            std = np.maximum(self.std, 1e-5)  # to avoid std -> 0
            x[i] = np.divide((x[i] - self.mean), std)  # normalization

        return x
