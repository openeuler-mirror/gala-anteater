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
"""
Time: 2023-6-12
Author: Zhenxing
Description: The normalization method for the data processing
"""

import os

import joblib
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from anteater.utils.log import logger


class Normalization:
    """The normalization class for the data processing"""

    filename = "normalization.pkl"

    def __init__(self, **kwargs):
        """The normalizer initializer"""
        self.normalizer = MinMaxScaler(**kwargs)

    @classmethod
    def load(cls, folder, **kwargs):
        """Loads model from the file"""
        file = os.path.join(folder, cls.filename)

        if not os.path.isfile(file):
            logger.warning("Unknown model file, load default norm model!")
            return Normalization(**kwargs)

        model = cls(**kwargs)
        model.normalizer = joblib.load(file)
        return model

    @staticmethod
    def clip_transform(value, alpha=10, mean=None, std=None, is_clip=True):
        '''
            @params:
                value: [n, d], n: number of feature, d: feature vector.
                alpha: float, clip bias for std.
            @return:
                value: after transformer [n, d]
                mean: [n, 1] for test
                std: [n, 1] for test
        '''
        if mean is None:
            mean = np.mean(value, axis=-1)
        if std is None:
            std = np.std(value, axis=-1)
            # 标准差太小的地方，

        for i in range(value.shape[0]):
            # compute clip value: (mean - a * std, mean + a * std)
            if is_clip:
                clip_value = mean + alpha * std
                temp = value[i] < clip_value
                value[i] = temp * value[i] + (1 - temp) * clip_value
                clip_value = mean - alpha * std
                temp = value[i] > clip_value
                value[i] = temp * value[i] + (1 - temp) * clip_value

            # to avoid std -> 0
            std = np.maximum(std, 1e-5)
            # normalization
            value[i] = (value[i] - mean) / std

        return value, mean, std

    def save(self, folder):
        """Saves the model into the file"""
        model_path = os.path.join(folder, self.filename)
        joblib.dump(self.normalizer, model_path)

    def fit_transform(self, x):
        """Fits and transforms the data"""
        x_norm = self.normalizer.fit_transform(x)
        return x_norm

    def transform(self, x):
        """Transform the data"""
        x_norm = self.normalizer.transform(x)
        return x_norm
