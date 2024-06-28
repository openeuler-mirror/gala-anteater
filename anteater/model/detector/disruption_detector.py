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
import pandas as pd

from itertools import groupby
from typing import List, Tuple
from math import floor

from anteater.core.anomaly import Anomaly
from anteater.core.kpi import KPI, ModelConfig
from anteater.core.ts import TimeSeries
from anteater.model.detector.base import Detector
from anteater.source.metric_loader import MetricLoader
from anteater.model.algorithms.smooth import smoothing
from anteater.model.algorithms.n_sigma import n_sigma
from anteater.model.algorithms.normalization import Normalization
from anteater.model.algorithms.spot import Spot
from anteater.utils.common import divide, GlobalVariable
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger


class ContainerDisruptionDetector(Detector):
    def __init__(self, data_loader: MetricLoader, config: ModelConfig, **kwargs):
        """The detector base class initializer"""
        super().__init__(data_loader, **kwargs)
        self.config = config
        self.q = 1e-3
        self.level = 0.98
        self.smooth_win = 3

    def detect_kpis(self, kpis: List[KPI]):
        """Executes anomaly detection on kpis"""
        start, end = dt.last(minutes=1)
        machine_ids = self.get_unique_machine_id(start, end, kpis)
        anomalies = []
        for _id in machine_ids:
            for kpi in kpis:
                anomalies.extend(self.detect_signal_kpi(kpi, _id))

        return anomalies

    def detect_signal_kpi(self, kpi, machine_id: str) -> List[Anomaly]:
        """Detects kpi based on signal time series anomaly detection model"""
        anomalies = []
        anomalies_n_sigma = self.detect_by_n_sigma(kpi, machine_id)
        if anomalies_n_sigma:
            anomalies.extend(anomalies_n_sigma)

        anomalies_spot = self.detect_by_spot(kpi, machine_id)
        if anomalies_spot:
            anomalies.extend(anomalies_spot)

        return anomalies

    def detect_by_n_sigma(self, kpi, machine_id: str) -> List[Anomaly]:
        outlier_ratio_th = kpi.params['outlier_ratio_th']
        ts_scores = self.cal_n_sigma_score(
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
                details={'event_source': 'n-sigma'})
            for _ts, _score in ts_scores
        ]

        return anomalies

    def cal_n_sigma_score(self, metric, machine_id: str, **kwargs) \
            -> List[Tuple[TimeSeries, int]]:
        """Calculates metrics' ab score based on n-sigma method"""
        method = kwargs.get('method', 'abs')
        look_back = kwargs.get('look_back')
        smooth_params = kwargs.get('smooth_params')
        obs_size = kwargs.get('obs_size')
        n = kwargs.get('n', 3)
        if GlobalVariable.is_test_model:
            start, end = GlobalVariable.start_time, GlobalVariable.end_time
        else:
            start, end = dt.last(minutes=look_back)
        point_count = self.data_loader.expected_point_length(start, end)
        ts_list = self.data_loader.get_metric(
            start, end, metric, machine_id=machine_id)
        ts_scores = []
        outlier = []
        for _ts in ts_list:
            dedup_values = [k for k, g in groupby(_ts.values)]
            if sum(_ts.values) == 0 or \
               len(_ts.values) < point_count * 0.6 or \
               len(_ts.values) > point_count * 1.5 or \
               all(x == _ts.values[0] for x in _ts.values):
                score = 0
            elif len(dedup_values) < point_count * 0.6:
                score = 0
            else:
                smoothed_val = smoothing(_ts.values, **smooth_params)
                outlier, _, _ = n_sigma(
                    smoothed_val, obs_size=obs_size, n=n, method=method)
                score = divide(len(outlier), obs_size)
            print('data: ', _ts.values)
            print('n-sigma result: ', outlier)
            logger.info('n-sigma detected: %d , total: %d , metric: %s, image: %s',
                           score*obs_size, obs_size, _ts.metric, _ts.labels['container_image'])

            ts_scores.append((_ts, score))

        return ts_scores

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
                details={'event_source': 'spot'})
            for _ts, _score in ts_scores
        ]

        return anomalies

    def cal_spot_score(self, metric, machine_id: str, **kwargs) \
            -> List[Tuple[TimeSeries, int]]:
        """Calculates metrics' ab score based on n-sigma method"""
        method = kwargs.get('method', 'abs')
        look_back = kwargs.get('look_back')
        smooth_params = kwargs.get('smooth_params')
        obs_size = kwargs.get('obs_size')
        n = kwargs.get('n', 3)
        if GlobalVariable.is_test_model:
            start, end = GlobalVariable.start_time, GlobalVariable.end_time
        else:
            start, end = dt.last(minutes=look_back)
        point_count = self.data_loader.expected_point_length(start, end)
        ts_list = self.data_loader.get_metric(
            start, end, metric, machine_id=machine_id)
        ts_scores = []
        for _ts in ts_list:
            ts_series = pd.Series(_ts.values)
            ts_series_list = self._check_bound_type('bi_bound', ts_series)
            result = np.zeros((obs_size,), dtype=np.int32)
            for _ts_series in ts_series_list:
                _ts_series_train = _ts_series[:-obs_size]
                _ts_series_test = _ts_series[-obs_size:]

                # fit model
                smooth_win = self._check_smooth(_ts_series_train)
                _ts_series_train = _ts_series_train.rolling(window=smooth_win).mean().bfill().ffill().values

                if self._is_peak_empty(_ts_series_train):
                    if np.max(_ts_series_train) == 0:
                        noise_data = np.random.normal(0, scale=1e-6, size=_ts_series_train.shape)
                    else:
                        noise_ratio = np.random.randint(-1e5, 1e5, size=_ts_series_train.shape) / 1e6
                        noise_data = noise_ratio * _ts_series_train
                    _ts_series_train = noise_data + _ts_series_train
                    logger.warning("peak data are same.")

                _ts_series_train, mean, std = Normalization.clip_transform(
                    _ts_series_train[np.newaxis, :], is_clip=False)
                _ts_series_train = _ts_series_train[0]
                spot = Spot(q=self.q)
                level = self._check_level(_ts_series_train, self.level)
                spot.initialize(_ts_series_train, level=level)

                # predict
                smooth_win = self._check_smooth(_ts_series_test)
                _ts_series_test = _ts_series_test.rolling(window=smooth_win).mean().bfill().ffill().values
                _ts_series_test, _, _ = Normalization.clip_transform(
                    _ts_series_test[np.newaxis, :], mean=mean, std=std, is_clip=False)
                _ts_series_test = _ts_series_test[0]
                thr_with_alarms = spot.run(_ts_series_test, with_alarm=True)
                bound_result = np.array(_ts_series_test > thr_with_alarms["thresholds"], dtype=np.int32)
                result += bound_result
            output = np.sum(result)
            print('data: ', _ts.values)
            print('spot result: ', result)
            logger.warning('spot detected: %d , total: %d , metric: %s, image: %s',
                           output, obs_size, _ts.metric, _ts.labels['container_image'])

            score = divide(output, obs_size)
            ts_scores.append((_ts, score))

        return ts_scores

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
        peak_num = int(level*data_size)

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

