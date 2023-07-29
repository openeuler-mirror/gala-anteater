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

import pandas as pd
from pandas import DataFrame


class ClipScaler:
    """Clipping and scaling features by a given range from the std

    Formula:
        1. x = min(max(mean - a * std, x)  mean + a * std)
        2. x = (x - mean) / std
    """
    def __init__(self, alpha=20) -> None:
        self.alpha = alpha

        self.mean = pd.Series()
        self.std = pd.Series()

    def fit(self, x):
        """Compute the mean and std to be used for later scaling"""
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0).map(lambda v: 1 if v < 1e-4 else v)

        return self

    def fit_transform(self, x):
        """Compute and scale the feature"""
        return self.fit(x).transform(x)

    def transform(self, x: DataFrame):
        """Scale the features by using computed mean and std"""
        if self.mean.empty or self.std.empty:
            raise ValueError('Empty mean and std of scaler')

        # compute clip value: (mean - a * std, mean + a * std)
        upper = self.mean + self.alpha * self.std
        lower = self.mean - self.alpha * self.std
        x = x.clip(upper=upper, lower=lower, axis=1)

        # normalization
        x = (x - self.mean).divide(self.std)

        return x
