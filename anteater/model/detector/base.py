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
from abc import abstractmethod
from typing import List

from anteater.core.anomaly import Anomaly
from anteater.core.kpi import KPI, Feature, JobConfig
from anteater.core.time_series import TimeSeriesScore
from anteater.model.algorithms.spectral_residual import SpectralResidual
from anteater.model.algorithms.slope import check_trend
from anteater.source.metric_loader import MetricLoader
from anteater.utils.common import same_intersection_key_value
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger
from anteater.utils.timer import timer


class Detector:
    """The kpi anomaly detector base class"""

    def __init__(self, data_loader: MetricLoader):
        """The detector base class initializer"""
        self.data_loader = data_loader

    @abstractmethod
    def detect_kpis(self, kpis: List[KPI]) -> List[Anomaly]:
        """Executes anomaly detection on kpis"""
        pass

    def get_unique_machine_id(self, start, end, kpis: List[KPI]) -> List[str]:
        """Gets unique machine ids during past minutes"""
        metrics = [_kpi.metric for _kpi in kpis]
        machine_ids = self.data_loader.get_unique_machines(start, end, metrics)
        return machine_ids

    def execute(self, job_config: JobConfig) -> List[Anomaly]:
        """The main function of the detector"""
        kpis = job_config.kpis
        features = job_config.features
        n = job_config.root_cause_number
        if not kpis:
            logger.info(f"Empty kpis in detector: {self.__class__.__name__}!")
            return []

        return self._execute(kpis, features, top_n=n)

    def find_root_causes(self, anomalies: List[Anomaly], features: List[Feature], top_n=3)\
            -> List[Anomaly]:
        """Finds root causes for each anomaly events"""
        result = []
        for anomaly in anomalies:
            cause_features = self.recommend_cause_features(
                features, anomaly.machine_id, filters=anomaly.labels, top_n=top_n)
            anomaly.root_causes = cause_features
            result.append(anomaly)

        return result

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
                    _ts_score.score = 0
                if same_intersection_key_value(_ts_score.ts.labels, filters):
                    ts_scores.append(_ts_score)

        ts_scores = [v for v in ts_scores if v.score > 0]
        ts_scores = sorted(ts_scores, key=lambda x: x.score, reverse=True)
        ts_scores = ts_scores[: top_n]
        ts_scores = sorted(ts_scores, key=lambda x: metric_priority[x.ts.metric])

        return ts_scores

    def cal_kpi_anomaly_score(self, anomalies: List[Anomaly], kpis: List[KPI]):
        """Calculates anomaly scores for the anomaly kpis"""
        kpi_atrends = {k.metric: k.atrend for k in kpis}
        for anomaly in anomalies:
            ts_scores = self.cal_anomaly_score(anomaly.metric, description="", machine_id=anomaly.machine_id)
            for _ts_s in ts_scores:
                if same_intersection_key_value(_ts_s.ts.labels, anomaly.labels):
                    if not check_trend(_ts_s.ts.values, kpi_atrends[anomaly.metric]):
                        anomaly.score = 0
                    else:
                        anomaly.score = _ts_s.score
                break

        return anomalies

    def cal_anomaly_score(
            self,
            metric: str,
            description: str,
            machine_id: str)\
            -> List[TimeSeriesScore]:
        """Calculates metric anomaly scores based on sr model"""
        start, end = dt.last(minutes=6)
        point_count = self.data_loader.expected_point_length(start, end)
        model = SpectralResidual(12, 24, 50)
        ts_scores = []
        ts_list = self.data_loader.\
            get_metric(start, end, metric, label_name='machine_id', label_value=machine_id)
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

    @timer
    def _execute(self, kpis: List[KPI], features: List[Feature], **kwargs) -> List[Anomaly]:
        logger.info(f"Execute model: {self.__class__.__name__}!")
        anomalies = self.detect_kpis(kpis)
        if anomalies:
            logger.info(f'{len(anomalies)} anomalies was detected on {self.__class__.__name__}.')
            anomalies = self.find_root_causes(anomalies, features, **kwargs)
            anomalies = self.cal_kpi_anomaly_score(anomalies, kpis)

        return anomalies
