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
from typing import Dict, List, Tuple
from abc import ABC
from datetime import datetime
import pytz
from math import floor
from itertools import groupby

import requests
import numpy as np
import pandas as pd

from anteater.core.ts import TimeSeries
from anteater.core.anomaly import Anomaly, RootCause
from anteater.core.kpi import KPI, ModelConfig, Feature
from anteater.utils.common import divide, GlobalVariable
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.timer import timer
from anteater.utils.log import logger
from anteater.source.metric_loader import MetricLoader
from anteater.model.detector.base import Detector
from anteater.model.algorithms.n_sigma import n_sigma_ex
from anteater.model.algorithms.normalization import Normalization
from anteater.model.algorithms.spot import Spot
from anteater.model.algorithms.ts_dbscan import TSDBSCAN


class ContainerDisruptionDetector(Detector):
    def __init__(self, data_loader: MetricLoader, config: ModelConfig, **kwargs):
        """The detector base class initializer"""
        super().__init__(data_loader, **kwargs)
        self.config = config
        self.q = 1e-3
        self.level = 0.98
        self.smooth_win = 3
  
        self.container_num = 0
        self.start_time = None
        self.end_time = None
 
    @timer
    def _execute(self, kpis: List[KPI], features: List[Feature], **kwargs) \
            -> List[Anomaly]:
        logger.info('Execute cdt model: %s.', self.__class__.__name__)
        anomalies = self.detect_and_rca(kpis)

        return anomalies

    def detect_and_rca(self, kpis: List[KPI]):
        start, end = dt.last(minutes=20)
        machine_ids = self.get_unique_machine_id(start, end, kpis)
        anomalies = []
        for _id in machine_ids:
            for kpi in kpis:
                anomalies.extend(self.detect_signal_kpi(kpi, _id))

        logger.info('total machine number is %d, container number is %d .',
                    len(machine_ids), self.container_num)
        self.container_num = 0

        return anomalies

    def detect_signal_kpi(self, kpi, machine_id: str) -> List[Anomaly]:
        """Detects kpi based on signal time series anomaly detection model"""

        anomalies = []
        anomalies_spot = self.detect_by_spot(kpi, machine_id)
        if anomalies_spot:
            anomalies.extend(anomalies_spot)

        return anomalies

    def get_kpi_ts_list(self, metric, machine_id: str, look_back):

        if GlobalVariable.is_test_model:
            start_time, end_time = GlobalVariable.start_time, GlobalVariable.end_time
            self.start_time = start_time
            self.end_time = end_time

            ts_list = self.data_loader.get_metric(
                start_time, end_time, metric, machine_id=machine_id)
            point_count = self.data_loader.expected_point_length(strat_time, end_time)

        else:
            start, end = dt.last(minutes=look_back)
            self.start_time = start
            self.end_time = end

            point_count = self.data_loader.expected_point_length(start, end)
            ts_list = self.data_loader.get_metric(
                start, end, metric, machine_id=machine_id)

        return point_count, ts_list

    def detect_by_spot(self, kpi, machine_id: str) -> List[Anomaly]:
        outlier_ratio_th = kpi.params['outlier_ratio_th']
        ts_scores = self.cal_spot_score(
            kpi.metric, machine_id, **kpi.params)
        if not ts_scores:
            logger.warning('Key metric %s is null on the target machine %s!',
                           kpi.metric, machine_id)
            return []

        ts_scores = [t for t in ts_scores if t[1] >= outlier_ratio_th]
        anomalies = [
            Anomaly(
                machine_id=machine_id,
                metric=_ts.metric,
                labels=_ts.labels,
                score=float(_score),
                entity_name=kpi.entity_name,
                root_causes=_root_causes,
                details={'event_source': 'spot', 'info': _extra_info})
            for _ts, _score, _extra_info, _root_causes in ts_scores
        ]

        return anomalies

    def cal_spot_score(self, metric, machine_id: str, **kwargs) \
            -> List[Tuple[TimeSeries, int, Dict, List[RootCause]]]:
        """Calculates metrics' ab score based on n-sigma method"""
        look_back = kwargs.get('look_back')
        obs_size = kwargs.get('obs_size')
        ts_dbscan_detector = TSDBSCAN(kwargs)

        point_count, ts_list = self.get_kpi_ts_list(metric, machine_id, look_back)
        ts_scores = []
        root_causes = []
        extra_info = {}
        logger.info('machine %s, total detected %d containers.',
                    machine_id, len(ts_list))
        self.container_num += len(ts_list)
        for _ts in ts_list:
            # import pdb;pdb.set_trace()
            detect_result = ts_dbscan_detector.detect(_ts.values)
            if len(detect_result) != len(_ts.values):
                raise ""
            train_data = [_ts.values[i] for i in range(len(detect_result)) if detect_result[i] == 0]
            test_data = _ts.values[-obs_size:]
            dedup_values = [k for k, g in groupby(test_data)]
            train_dedup_values = [k for k, g in groupby(train_data)]
            if sum(_ts.values) == 0 or \
                    np.max(_ts.values) < 1e3 or \
                    len(_ts.values) < point_count * 0.6 or \
                    len(_ts.values) > point_count * 1.5 or \
                    all(x == _ts.values[0] for x in _ts.values) or\
                    len(dedup_values) < obs_size * 0.8 or \
                    len(train_dedup_values) < len(train_data) * 0.8:
                score = 0
            else:
                ts_series = pd.Series(_ts.values)
                ts_series_train = pd.Series(train_data)
                ts_series_list = self._check_bound_type('upper_bound', ts_series)
                ts_series_train_list = self._check_bound_type('upper_bound', ts_series_train)
                result = np.zeros((obs_size,), dtype=np.int32)
                for _ts_series, _ts_series_train in zip(ts_series_list, ts_series_train_list):
                    _ts_series_test = _ts_series[-obs_size:]
                    # fit model
                    # smooth_win = self._check_smooth(_ts_series_train)
                    # _ts_series_train = _ts_series_train.rolling(window=smooth_win).mean().bfill().ffill().values
                    _ts_series_train = _ts_series_train.values

                    noise_data = np.random.normal(0, scale=1e-6, size=_ts_series_train.shape)
                    _ts_series_train += noise_data
                    if self._is_peak_empty(_ts_series_train):
                        if np.max(_ts_series_train) != 0:
                            noise_ratio = np.random.randint(-1e5, 1e5, size=_ts_series_train.shape) / 1e6
                            noise_data = noise_ratio * _ts_series_train
                        _ts_series_train += noise_data
                        # logger.warning("peak data are same.")

                    _ts_series_train, mean, std = Normalization.clip_transform(
                        _ts_series_train[np.newaxis, :], is_clip=False)
                    _ts_series_train = _ts_series_train[0]
                    spot = Spot(q=self.q)
                    level = self._check_level(_ts_series_train, self.level)
                    spot.initialize(_ts_series_train, level=level)

                    # predict
                    # smooth_win = self._check_smooth(_ts_series_test)
                    # _ts_series_test = _ts_series_test.rolling(window=smooth_win).mean().bfill().ffill().values
                    _ts_series_test = _ts_series_test.values
                    _ts_series_test, _, _ = Normalization.clip_transform(
                        _ts_series_test[np.newaxis, :], mean=mean, std=std, is_clip=False)
                    _ts_series_test = _ts_series_test[0]
                    thr_with_alarms = spot.run(_ts_series_test, with_alarm=True)
                    bound_result = np.array(_ts_series_test > thr_with_alarms["thresholds"], dtype=np.int32)
                    result += bound_result
                output = np.sum(result)
                if output >= 3:
                    print('data: ', _ts.values)
                    print('spot result: ', result)
                    container_hostname = _ts.labels.get('container_name', '')
                    machine_id = _ts.labels.get('machine_id', '')
                    logger.info('spot detected: %d , total: %d , metric: %s, host_name: %s',
                                output, obs_size, _ts.metric, container_hostname)
                    
                    extra_info = self.get_container_extra_info(machine_id, 
                                                               container_hostname, 
                                                               self.start_time, 
                                                               self.end_time, 
                                                               obs_size)
                    print('extra_info', extra_info)
                    root_causes = self.find_discruption_source(_ts, ts_list)

                score = divide(output, obs_size)
                
            ts_scores.append((_ts, score, extra_info, root_causes))

        return ts_scores

    def find_discruption_source(self, victim_ts: TimeSeries, all_ts: List[TimeSeries]) \
            -> List[RootCause]:
        root_causes = []
        tmp_causes = []
        for ts in all_ts:
            # container_hostname = ts.labels.get('container_hostname', '')
            # info = self.queryContainerInfo(container_hostname) if container_hostname else {}
            # cpu_num = info.get('cpu', 0)
            cpu_num = int(ts.labels.get('cpu_num', '0'))
            if ts is victim_ts and cpu_num < 5:
                continue

            agg_data = []
            for i in range(len(victim_ts.time_stamps)):
                data = {
                    "victim": victim_ts.values[i],
                    "source": ts.values[i]
                }
                agg_data.append(data)

            agg_data_df = pd.DataFrame(agg_data)
            self._normalize_df(agg_data_df)

            # metrics_correlation = agg_data_df.corr(method="spearman")
            # metrics_correlation = agg_data_df.corr(method="kendall")
            metrics_correlation = agg_data_df.corr(method="pearson")

            sorted_metrics_correlation = abs(metrics_correlation.iloc[0]).sort_values(ascending=False)
            # print("sorted_metrics_correlation:", sorted_metrics_correlation)
            
            causes = {
                'score': round(sorted_metrics_correlation.values[-1], 3),
                'cpu_num': cpu_num,
                'metric': ts.metric,
                'labels':ts.labels
            }
            causes['labels']['cpu_num'] = cpu_num

            if causes['score'] > 0.5:
                tmp_causes.append(causes)

            # root_causes.append(RootCause(
            #     metric=ts.metric,
            #     labels=ts.labels,
            #     score=round(sorted_metrics_correlation.values[-1], 3)))

        tmp_causes.sort(key=lambda x: (x['labels']['cpu_num'], x['score']), reverse=True)

        root_causes = [RootCause(metric=causes['metric'], labels=causes['labels'], score=causes['score'])
                       for causes in tmp_causes]

        # print("root_causes:", root_causes)

        return root_causes[:3]
        
    def get_container_extra_info(self, machine_id: str, 
                                       container_name: str, 
                                       start_time: datetime, 
                                       end_time: datetime, 
                                       obs_size: int) -> Dict:
        extra_metrics = self.config.params.get('extra_metrics', '').split(',')
        # print(extra_metrics)
        result = {'container_name': container_name, 
                  'machine_id': machine_id}
        for metric in extra_metrics:
            ts_list = self.data_loader.get_metric(start_time, end_time, metric, machine_id=machine_id)
            for _ts in ts_list:
                if container_name == _ts.labels.get('container_name', ''):
                    values = _ts.values
                    trend = self.cal_trend(values, obs_size)
                    result[metric] = trend
                    result['appkey'] = _ts.labels.get('appkey', '')
                    result['cpu_num'] = int(_ts.labels.get('cpu_num', '0'))
                    # print("***", container_name, metric, _ts.values, trend)
                    break
        return result


    @staticmethod
    def _normalize_df(df):
        cols = list(df)
        for item in cols:
            if df[item].dtype == 'int64' or df[item].dtype == 'float64':
                max_tmp = np.max(np.array(df[item]))
                min_tmp = np.min(np.array(df[item]))
                if max_tmp != min_tmp:
                    df[item] = df[item].apply(
                        lambda x: (x - min_tmp) * 1 / (max_tmp - min_tmp))

    @staticmethod
    def _check_bound_type(bound_type, metric_data):
        if bound_type == "bi_bound":
            data = metric_data, -metric_data
        elif bound_type == "lower_bound":
            data = -metric_data,
        else:
            data = metric_data,

        return data

    @staticmethod
    def _check_level(metric_data, level):
        data_size = len(metric_data)
        if int(data_size * (1 - level)) == 0:
            peak = 2
            level = 1.0 - peak / float(data_size) - 1e-6

        return level

    def _is_peak_empty(self, metric_data):
        data_size = len(metric_data)
        sort_data = np.sort(metric_data)
        level = self.level - floor(self.level)
        peak_num = int(level * data_size)

        if peak_num == 0:
            peak_num = min(2, data_size)

        init_threshold = sort_data[peak_num]
        peaks = metric_data[metric_data > init_threshold]

        return peaks.size == 0

    def _check_smooth(self, metric_data):
        data_size = len(metric_data)
        if data_size < self.smooth_win:
            smooth_win = data_size // 2
        else:
            smooth_win = self.smooth_win

        return smooth_win

    @staticmethod
    def cal_trend(metric_values: list, obs_size: int) -> float:
        pre = metric_values[:-obs_size]
        check = metric_values[-obs_size:]
        pre_mean = np.mean(pre)
        check_mean = np.mean(check)
        trend = (check_mean - pre_mean) / pre_mean if pre_mean > 0 else 0.0
        # print('trend: ', round(trend, 3), pre_mean, check_mean, '|||', pre, check)
        
        return round(trend, 3)
