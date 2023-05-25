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
import math

import numpy as np
import pandas as pd
import collections

from scipy.stats import pearsonr
from pandas import DataFrame
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

from anteater.utils.log import logger


class FeaturePortrait(object):
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_time_sec = None
        self.end_time_sec = None

        self.data_max = None
        self.data_min = None

        self.sample_count = 0

        self.data_df = pd.DataFrame()

        self.data_set = []

        self.label_set = 0
        self.label_set_normal = 0
        self.label_set_abnormal = 0
        self.sample_rate = []
        self.sample_missing_rate = 0

    def get_single_metric_portrait(self, df: DataFrame):
        """
        feature portrait entry point

        :param df: time series dataframe
        Examples: df.head(3)
                                     metric name
        2023-05-23 07:44:27              10.0
        2023-05-23 07:44:32              20.0
        2023-05-23 07:44:37              30.0

        :return: portrait result
        """
        df['ts'] = pd.to_datetime(df.index).view('int64') // 10 ** 9
        metric = df.columns[0]

        self.__calc_feature_type(df)
        self.__calc_sample_rate(df)
        self.__calc_missing_data_rate(df)

        portrait = {
            'feature_type': self.feature_type,
            'start_time': df.index[0].strftime('%Y-%m-%d %X'),
            'end_time': df.index[-1].strftime('%Y-%m-%d %X'),
            'sample_count': len(df),
            'sample_rate': int(np.median(self.sample_rate)) if len(self.sample_rate) else 60,
            'sample_missing_rate': self.sample_missing_rate,
            'mean': df[metric].mean(),
            'median': df[metric].median(),
            'min': df[metric].min(),
            'quantile_0.25': df[metric].quantile(0.25),
            'quantile_0.50': df[metric].quantile(0.5),
            'quantile_0.75': df[metric].quantile(0.75),
            'max': df[metric].max(),
            'var': df[metric].var(),
            'std': df[metric].std()
        }

        return portrait

    def get_multi_metric_relevance(self, df: DataFrame, key_metric: str, threshold=None, topk=None):
        """
        calculate multi metrics's relevance to key metric

        :param df:
        Examples: df.head(3)
                                   metric_name_1  ...  metric_name_n
        2023-05-25 10:42:52              2.0      ...      0.3
        2023-05-25 10:42:57              5.0      ...      0.7
        2023-05-25 10:43:02              4.0      ...      0.1
        :param key_metric: the relevance of metric
        :param threshold: the relevance threshold of the key metric
        :param topk: the top k relevant metric of the key metric
        :return: relevant_result : collections.defaultdict(list)
        """
        print(df.head(3))
        if (threshold and topk) or (not threshold and not topk):
            raise ValueError("threshold and top k can't be set or not set in the same time")
        relevant_result = collections.defaultdict(list)
        relevant_score = []

        for col in df.columns:
            correlation_coef = pearsonr(df[key_metric].fillna(0), df[col].fillna(0))[0]
            relevant_score.append({
                'name': col,
                'score': round(correlation_coef, 3) if not math.isnan(correlation_coef) else 0
            })
        relevant_score.sort(key=lambda x: -abs(x['score']))

        if threshold is not None:
            for item in relevant_score:
                if item['score'] >= threshold:
                    relevant_result["relevant"].append(item)
                else:
                    relevant_result["irrelevant"].append(item)
        elif topk is not None:
            for item in relevant_score:
                if len(relevant_result["relevant"]) < topk:
                    relevant_result["relevant"].append(item)
                else:
                    relevant_result["irrelevant"].append(item)

        return relevant_result

    def __calc_sample_rate(self, df: DataFrame):
        """Calculate data sample rate"""
        last_ts = None
        for ts in df['ts']:
            if last_ts is not None:
                self.sample_rate.append(ts - last_ts)
            last_ts = ts

    def __calc_missing_data_rate(self, df: DataFrame):
        """Calculate data missing rate"""
        metric = df.columns[0]
        missing_data = df[metric].isna().sum()
        if len(df) > 0:
            self.sample_missing_rate = float(missing_data) / len(df)

    def __calc_feature_type(self, df: DataFrame):
        """Calculate data feature type"""
        ts = df.columns[1]
        metric = df.columns[0]

        m = np.asarray(df[metric])
        if (m == m[0]).all():
            self.feature_type = 'constant'
        elif abs(pearsonr(df[ts], df[metric])[0]) > 0.99:
            self.feature_type = 'linear'
        elif self.__check_white_noise(df[metric]):
            self.feature_type = 'white_noise'
        else:
            self.feature_type = 'normal'

    @staticmethod
    def __check_white_noise(series) -> bool:
        """Check dats is white noise or not"""
        p_value_adf = round(adfuller(series.values)[1], 2)
        p_value_ljbox = round(acorr_ljungbox(series.values, lags=1, boxpierce=True, return_df=False).iat[0, 1], 2)

        logger.debug(f"white noise check: p_value_adf(0.01)={p_value_adf}, p_value_ljbox(0.05)={p_value_ljbox}")

        return p_value_adf < 0.01 and p_value_ljbox >= 0.05
