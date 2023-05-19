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
import json
import os.path
import collections
import pandas as pd

from typing import List

from anteater.config import AnteaterConf
from anteater.core.time_series import TimeSeries
from anteater.feature_extraction.feature_portrait import FeaturePortrait
from anteater.source.metric_loader import MetricLoader
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger

ANTEATER_DATA_PATH = "D:\\project\\os\\gala-anteater\\"  # '/etc/gala-anteater/'


class FeatureExtract(object):
    def __init__(self, data_loader: MetricLoader, metric_list: []):
        self.data_loader = data_loader
        self.metric_list = metric_list
        self.feature_portrait = FeaturePortrait()

    def analyze(self):
        portrait_result = collections.defaultdict(list)

        for metric in self.metric_list:
            df = None
            for _time_series in self.__load_time_series(metric):
                df = _time_series.to_df()
                df['ts'] = pd.to_datetime(df.index).view('int64')//10**9
                break

            if df is not None:
                portrait_result[metric] = self.feature_portrait.get_portrait(df)

        logger.debug(portrait_result)

    def __load_time_series(self, metric) -> List[TimeSeries]:
        """Loads time series of the target kpi"""
        start, end = dt.last(minutes=1)
        time_series = self.data_loader.get_metric(start, end, metric, operator='avg', keys="machine_id")

        return time_series


if __name__ == '__main__':
    conf = AnteaterConf()
    conf.load_from_yaml(ANTEATER_DATA_PATH)
    loader = MetricLoader(conf)

    current_path = os.path.dirname(__file__)
    with open(os.path.join(current_path, 'feature_list.json')) as fl:
        metrics = json.load(fl)["features"]

    feature_extract = FeatureExtract(loader, metrics)
    feature_extract.analyze()
