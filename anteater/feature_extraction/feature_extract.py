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

from typing import List

from anteater.config import AnteaterConf
from anteater.core.ts import TimeSeries
from anteater.feature_extraction.feature_portrait import FeaturePortrait
from anteater.source.metric_loader import MetricLoader
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger

ANTEATER_DATA_PATH = "D:\\project\\os\\gala-anteater\\"  # '/etc/gala-anteater/'


class FeatureExtract(object):
    """The feature extract entry class.

    analyze method can portrait single metric and
    get relevance between key metric and other metrics.
    """
    def __init__(self, data_loader: MetricLoader, metric_list: []):
        self.data_loader = data_loader
        self.metric_list = metric_list
        self.fp = FeaturePortrait()

    def analyze(self, machine_id: str, key_metric: str):
        """
        Analyze metric' feature and relevance.

        :param machine_id: one machine's id information
        :param key_metric: base metric
        :return: single_metric_portrait : collections.defaultdict(list),
                 multi_metric_relevance : collections.defaultdict(list)
        """
        single_metric_portrait = collections.defaultdict(list)
        multi_metric_relevance = collections.defaultdict(list)

        multi_metric_df = None
        start, end = dt.last(minutes=60)
        # get single metric portrait
        for metric in self.metric_list:
            df = None
            for _time_series in self.__load_time_series(start, end, metric, machine_id):
                df = _time_series.to_df()
                break

            if df is not None:
                multi_metric_df = df if multi_metric_df is None else multi_metric_df.join(df)
                single_metric_portrait[metric] = self.fp.get_single_metric_portrait(df)

        # get multi metric relevance
        if multi_metric_df is not None:
            multi_metric_relevance = self.fp.get_multi_metric_relevance(multi_metric_df, key_metric, threshold=0.6)

        logger.debug(single_metric_portrait, multi_metric_relevance)
        return single_metric_portrait, multi_metric_relevance

    def __load_time_series(self, start, end, metric: str, machine_id: str) -> List[TimeSeries]:
        """
        Loads time series of the target machine from time of start to end.

        :param start: data of start time
        :param end: data of start time
        :param metric: metric name to load
        :param machine_id: the machine's id of the data to load
        :return: time_series: List[TimeSeries]
        """
        time_series = self.data_loader.get_metric(start,
                                                  end,
                                                  metric,
                                                  operator='avg',
                                                  keys="machine_id",
                                                  machine_id=machine_id)

        return time_series


if __name__ == '__main__':
    conf = AnteaterConf()
    conf.load_from_yaml(ANTEATER_DATA_PATH)
    loader = MetricLoader(conf)

    current_path = os.path.dirname(__file__)
    with open(os.path.join(current_path, 'feature_list.json')) as fl:
        metrics = json.load(fl)["features"]

    feature_extract = FeatureExtract(loader, metrics)
    m_id = '7c2fbaf8-4528-4aaf-90c1-5c4c46b06ebe'
    k_metric = 'gala_gopher_sli_rtt_nsec'
    feature_extract.analyze(m_id, k_metric)
