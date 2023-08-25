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

from datetime import datetime
import math
from abc import abstractmethod
from typing import List, Tuple

from anteater.core.anomaly import Anomaly, RootCause
from anteater.core.kpi import KPI, Feature, JobConfig
from anteater.core.ts import TimeSeries
from anteater.model.algorithms.spectral_residual import SpectralResidual
from anteater.model.algorithms.slope import check_trend
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

    def get_unique_machine_id(self, start: datetime, end: datetime,
                              kpis: List[KPI]) -> List[str]:
        """Gets unique machine ids during past minutes"""
        metrics = [_kpi.metric for _kpi in kpis]
        machine_ids = self.data_loader.get_unique_machines(start, end, metrics)
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
