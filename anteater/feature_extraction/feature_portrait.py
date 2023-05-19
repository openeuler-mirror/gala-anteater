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
from scipy.stats import pearsonr
from pandas import DataFrame
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

from anteater.utils.log import logger


class FeaturePortrait(object):

    def __init__(self):
        self.data_df = None

    def get_portrait(self, df: DataFrame):
        """
        feature portrait main function
        :param df: time series dataframe
        :return: portrait result
        """
        ts = df.columns[1]
        metric = df.columns[0]

        result = pearsonr(df[ts], df[metric])
        if np.isnan(result[0]):
            feature_type = 'invariant'
        elif abs(result[0]) > 0.99:
            feature_type = 'linear'
        elif self.__is_white_noise(df[metric]):
            feature_type = 'white_noise'
        else:
            feature_type = 'normal'

        portrait = {
            'feature_type': feature_type,
            'count': len(df),
            'max': df[metric].max(),
            'min': df[metric].min(),
            'mean': df[metric].mean(),
            'var': df[metric].var(),
            'std': df[metric].std()
        }

        return portrait

    def __is_white_noise(self, series, adf_threshold=0.01, box_threshold=0.05):
        p_value_adf = round(adfuller(series.values)[1], 2)
        p_value_ljbox = round(acorr_ljungbox(series.values, lags=1, boxpierce=True, return_df=False).iat[0, 1], 2)

        result = p_value_adf < adf_threshold and p_value_ljbox >= box_threshold
        logger.debug(f"white noise check: p_value_adf(0.01)={p_value_adf}, p_value_ljbox(0.05)={p_value_ljbox}")

        return result
