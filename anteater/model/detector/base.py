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

import math
import pandas as pd
import numpy as np
from functools import reduce
from abc import abstractmethod
from typing import List, Tuple
from datetime import datetime

from anteater.core.anomaly import Anomaly, RootCause
from anteater.core.kpi import KPI, Feature, JobConfig
from anteater.core.ts import TimeSeriesScore, TimeSeries
from anteater.model.algorithms.spectral_residual import SpectralResidual
from anteater.model.algorithms.slope import check_trend
from anteater.model.algorithms.pearson import pearson_correlation
from anteater.source.metric_loader import MetricLoader
from anteater.utils.common import same_intersection_pairs
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger
from anteater.utils.timer import timer


class Detector:
    """The kpi anomaly detector base class"""

    def __init__(self, data_loader: MetricLoader, **kwargs) -> None:
        """The detector base class initializer"""
        self.data_loader = data_loader
        self.metric_similar = {}

    @staticmethod
    def get_timestamp(start, end, interval=5):
        """Gets a sequence of timestamps for a given time interval from start time to end time"""
        start_timestamp = datetime.timestamp(start)
        end_timestamp = datetime.timestamp(end)
        timestamp_list = np.arange(
            start_timestamp, end_timestamp + interval, interval)
        datetime_list = [datetime.strftime(datetime.utcfromtimestamp(
            timestamp), "%Y-%m-%d %H:%M:%S") for timestamp in timestamp_list]
        timestamp = pd.to_datetime(np.asarray(
            timestamp_list).astype(float) * 1000, unit="ms")
        index = pd.to_datetime(timestamp)
        series = pd.Series(datetime_list, index=index, name="timestamp")
        series = series[~series.index.duplicated()]

        return series.to_frame()

    @abstractmethod
    def detect_kpis(self, kpis: List[KPI]) -> List[Anomaly]:
        """Executes anomaly detection on kpis"""

    def execute(self, job_config: JobConfig) -> List[Anomaly]:
        """The main function of the detector"""
        kpis = job_config.kpis
        features = job_config.features
        n = job_config.root_cause_num

        if not kpis:
            logger.info('Empty kpi in detector: %s.',
                        self.__class__.__name__)
            return []

        return self._execute(kpis, features, top_n=n)

    def get_machines_to_devices(self, start: datetime, end: datetime,
                              kpis: List[KPI]) -> List[str]:
        """Gets unique machine ids during past minutes"""
        metrics = [_kpi.metric for _kpi in kpis]
        machines_to_devices_id = self.data_loader.get_topo(start, end, metrics, label_name="id")

        return machines_to_devices_id

    def get_unique_machine_id(self, start: datetime, end: datetime,
                              kpis: List[KPI]) -> List[str]:
        """Gets unique machine ids during past minutes"""
        metrics = [_kpi.metric for _kpi in kpis]
        machine_ids = self.data_loader.get_unique_machines(start, end, metrics)
        return machine_ids

    def get_unique_pod_id(self, start, end, kpis: List[KPI]) -> List[str]:
        """Gets unique pod ids during past minutes"""
        metrics = [_kpi.metric for _kpi in kpis]
        machine_ids = self.data_loader.get_unique_pods(start, end, metrics)
        return machine_ids

    def find_root_causes(self, anomalies: List[Anomaly],
                         features: List[Feature], top_n=3) -> List[Anomaly]:
        """Finds root causes for each anomaly events"""
        result = []
        for anomaly in anomalies:
            root_causes = self.cal_top_rac(anomaly, features, top_n=top_n)
            anomaly.root_causes = root_causes
            result.append(anomaly)

        return result

    def cal_top_rac(self, anomaly: Anomaly,
                    features: List[Feature], top_n=3) -> List[RootCause]:
        """calculates the top n root causes for the anomaly events"""
        root_causes = []
        for f in features:
            ts_scores = self.cal_metric_ab_score(f.metric, anomaly.machine_id)
            for _ts, _score in ts_scores:
                if not check_trend(_ts.values, f.atrend):
                    logger.info('Trends Filtered: %s', f.metric)
                    break

                if same_intersection_pairs(_ts.labels, anomaly.labels):
                    root_causes.append(RootCause(
                        metric=_ts.metric,
                        labels=_ts.labels,
                        score=_score))

        priorities = {f.metric: f.priority for f in features}
        root_causes.sort(key=lambda x: x.score, reverse=True)
        root_causes = root_causes[: top_n]
        root_causes.sort(key=lambda x: priorities[x.metric])

        return root_causes

    def recommend_cause_features(
            self,
            features: List[Feature],
            machine_id: str,
            filters: dict,
            top_n: int = 3)\
            -> List[TimeSeriesScore]:
        """Recommend cause features for any abnormal kpi"""
        metric_priority = {f.metric: f.priority for f in features}
        ts_scores = []
        for f in features:
            tmp_ts_scores = self.cal_anomaly_score(f.metric, f.description, machine_id=machine_id)
            for _ts_score in tmp_ts_scores:
                if not check_trend(_ts_score.ts.values, f.atrend):
                    logger.info(f"Trends Filtered: {f.metric}")
                    _ts_score.score = 0
                if same_intersection_pairs(_ts_score.ts.labels, filters):
                    ts_scores.append(_ts_score)

        ts_scores = [v for v in ts_scores if v.score > 0]
        ts_scores = sorted(ts_scores, key=lambda x: x.score, reverse=True)
        ts_scores = ts_scores[: top_n]
        ts_scores = sorted(ts_scores, key=lambda x: metric_priority[x.ts.metric])

        return ts_scores

    def cal_anomaly_score(
            self,
            metric: str,
            description: str,
            machine_id: str)\
            -> List[TimeSeriesScore]:
        """Calculates metric anomaly scores based on sr model"""
        start, end = dt.last(minutes=10)
        point_count = self.data_loader.expected_point_length(start, end)
        model = SpectralResidual(12, 24, 50)
        ts_scores = []
        ts_list = self.data_loader.get_metric(start, end, metric, machine_id=machine_id)
        for _ts in ts_list:
            if sum(_ts.values) == 0 or \
               len(_ts.values) < point_count * 0.9 or\
               len(_ts.values) > point_count * 1.5 or \
               all(x == _ts.values[0] for x in _ts.values):
                score = 0
            else:
                score = model.compute_score(_ts.values)
                score = max(score[-25:])

            if math.isnan(score) or math.isinf(score):
                score = 0

            ts_scores.append(TimeSeriesScore(ts=_ts, score=score, description=description))

        return ts_scores

    def get_usad_root_causes(self, anomalies: List[Anomaly], features: List[Feature], anomaly_scores, key="machine_id")\
            -> List[Anomaly]:
        """Finds root causes for each anomaly events in the Usad model"""
        result = []
        for anomaly in anomalies:
            cause_features = self.get_usad_cause_features(
                features, anomaly.machine_id, filters=anomaly.labels, anomaly_scores=anomaly_scores, key=key)

            anomaly.root_causes = cause_features
            result.append(anomaly)

        return result

    def cal_usad_kpi_anomaly_score(self, anomalies: List[Anomaly], anomaly_scores) -> List[Anomaly]:
        """Calculates anomaly scores for the anomaly kpis in the Usad model"""
        for anomaly in anomalies:
            if "@" in anomaly.metric:
                anomaly.score = float(
                    anomaly_scores[anomaly.metric.split("@")[0]])
            else:
                anomaly.score = float(anomaly_scores[anomaly.metric])

        return anomalies

    def get_usad_cause_features(
            self,
            features: List[Feature],
            machine_id: str,
            filters: dict,
            anomaly_scores, key="machine_id")\
            -> List[TimeSeriesScore]:
        """Recommend cause features for any abnormal kpi in the Usad model"""
        metric_priority = {f.metric: f.priority for f in features}
        ts_scores = []
        for f in features:
            tmp_ts_scores, metric_similar = self.cal_usad_anomaly_score(
                f.metric, f.description, machine_id=machine_id, score=float(anomaly_scores[f.metric]), top_n=1, key=key)
            if tmp_ts_scores:
                for _ts_score in tmp_ts_scores:
                    _ts_score.score = _ts_score.score if check_trend(_ts_score.ts.values, f.atrend) else 0
                    ts_scores.append(_ts_score)

        # 分数异常，复值等于
        ts_scores = [v for v in ts_scores if v.score >= 0]
        ts_scores = sorted(ts_scores, key=lambda x: x.score, reverse=True)

        if "@" in ts_scores[0].ts.metric:
            ts_scores = sorted(
                ts_scores, key=lambda x: metric_priority[x.ts.metric.split("@")[0]])
        else:
            ts_scores = sorted(
                ts_scores, key=lambda x: metric_priority[x.ts.metric])

        return ts_scores

    def cal_similarity(
            self,
            start,
            end,
            standard_series,
            target_metric: str,
            ts_list)\
            -> List[TimeSeries]:
        """Calculates the similarity between source and target metrics"""

        timestamp_df = self.get_timestamp(start, end)

        result_ts_list = TimeSeries(target_metric, {}, [], [])

        single_metric_dfs = [timestamp_df]
        single_metrics_list = []
        single_metrics_set = set()
        if len(ts_list):
            for _ts in ts_list:
                df = _ts.to_df()
                column_name = df.columns[0]

                if column_name not in single_metrics_set:
                    single_metrics_set.add(column_name)
                    single_metrics_list.append(column_name)
                    single_metric_dfs.append(df)

            single_df = pd.concat(single_metric_dfs, axis=1, join='outer')
            single_df = single_df.fillna(0)

            record = pearson_correlation(
                target_metric, single_df, standard_series, top_n=1)

            metrics = record[target_metric]
            self.metric_similar[target_metric] = metrics[0][0]

            idx = single_metrics_list.index(metrics[0][0])
            result_ts_list = ts_list[idx]

        return result_ts_list, self.metric_similar

    def cal_kpi_anomaly_score(self, anomalies: List[Anomaly],
                              kpis: List[KPI]) -> List[Anomaly]:
        """Calculates anomaly scores for the anomaly kpis"""
        atrends = {k.metric: k.atrend for k in kpis}
        for _anomaly in anomalies:
            metric = _anomaly.metric
            machine_id = _anomaly.machine_id
            labels = _anomaly.labels

            ts_scores = self.cal_metric_ab_score(metric, machine_id)
            for _ts, _score in ts_scores:
                if not same_intersection_pairs(_ts.labels, labels):
                    continue

                if not check_trend(_ts.values, atrends[metric]):
                    logger.info('Trends Filtered: %s', metric)
                    _anomaly.score = 0
                else:
                    _anomaly.score = _score

                break

        return anomalies

    def cal_usad_anomaly_score(self, metric: str, description: str,
                               machine_id: str, score: float, top_n=1, key="machine_id") -> List[TimeSeriesScore]:
        """Calculates metric anomaly scores in the Usad model"""
        ts_scores = []
        metric_similar = {}
        start, end = dt.last(minutes=10)

        timestamp_df = self.get_timestamp(start, end)
        standard_dfs = [timestamp_df]

        if key == "machine_id":
            ts_standard_list = self.data_loader.get_metric(start, end, metric, operator='avg', keys="machine_id", machine_id=machine_id)
        else:
            ts_standard_list = self.data_loader.get_metric(start, end, metric, operator='avg', keys="pod_id", pod_id=machine_id)

        if ts_standard_list:
            if len(ts_standard_list) > 1:
                raise ValueError(
                    f'Got multiple time_series based on machine id: {len(ts_standard_list)}')
            if ts_standard_list:
                standard_series = ts_standard_list[0]
            else:
                standard_series = TimeSeries(metric, {}, [], [])

            standard_df = standard_series.to_df()
            standard_dfs.append(standard_df)
            df = reduce(lambda left, right: pd.DataFrame(left).join(right, how='outer'), standard_dfs)
            df = df.fillna(0)
            standard_series = df[metric]
            if key == "machine_id":
                ts_list = self.data_loader.get_single_metric(
                    start, end, metric, keys="machine_id", machine_id=machine_id)
            else:
                ts_list = self.data_loader.get_single_metric(start, end, metric, keys="pod_id", pod_id=machine_id)

            ts_list, metric_similar = self.cal_similarity(start, end, standard_series, metric, ts_list)
            if ts_list:
                ts_scores.append(TimeSeriesScore(ts=ts_list, score=score, description=description))

        return ts_scores, metric_similar

    def cal_metric_ab_score(self, metric: str, machine_id: str) \
            -> List[Tuple[TimeSeries, int]]:
        """Calculates metric abnormal scores based on sr model"""
        start, end = dt.last(minutes=10)
        ts_list = self.data_loader.get_metric(
            start, end, metric, machine_id=machine_id)
        point_count = self.data_loader.expected_point_length(start, end)
        model = SpectralResidual(12, 24, 50)
        ts_scores = []
        for _ts in ts_list:
            if sum(_ts.values) == 0 or \
               len(_ts.values) < point_count * 0.9 or\
               len(_ts.values) > point_count * 1.5 or \
               all(x == _ts.values[0] for x in _ts.values):
                score = 0
            else:
                score = model.compute_score(_ts.values)
                score = max(score[-25:])

            if math.isnan(score) or math.isinf(score):
                score = 0

            ts_scores.append((_ts, score))

        return ts_scores

    @timer
    def _execute(self, kpis: List[KPI], features: List[Feature], **kwargs) \
            -> List[Anomaly]:
        logger.info('Execute model: %s.', self.__class__.__name__)
        anomalies = self.detect_kpis(kpis)
        if anomalies:
            logger.info('%d anomalies was detected on %s.',
                        len(anomalies), self.__class__.__name__)
            anomalies = self.find_root_causes(anomalies, features, **kwargs)
            anomalies = self.cal_kpi_anomaly_score(anomalies, kpis)

        return anomalies
