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
"""
Time:
Author:
Description: The anomaly detector implementation on APP Sli
"""

import math
from typing import List

from anteater.core.anomaly import Anomaly
from anteater.model.algorithms.spectral_residual import SpectralResidual
from anteater.model.smoother import conv_smooth
from anteater.model.three_sigma import three_sigma
from anteater.module.detector import Detector
from anteater.source.anomaly_report import AnomalyReport
from anteater.source.metric_loader import MetricLoader
from anteater.template.app_anomaly_template import AppAnomalyTemplate
from anteater.utils.common import divide
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger


class APPSliDetector(Detector):
    """APP sli detector while detects the abnormal
    on the sli metrics including rtt, tps, etc.
    """

    def __init__(self, data_loader: MetricLoader, anomaly_report: AnomalyReport):
        file_name = 'app_sli_rtt.json'
        super().__init__(data_loader, anomaly_report, file_name)

    def execute_detect(self, machine_id: str):
        for kpi in self.kpis:
            if kpi.kpi_type == 'rtt':
                anomalies = self.detect_rtt(kpi, machine_id)
            else:
                anomalies = self.detect_tps(kpi, machine_id)

            for anomaly in anomalies:
                self.report(anomaly, machine_id)

    def detect_rtt(self, kpi, machine_id: str) -> List[Anomaly]:
        """Detects rtt by rule-based model"""
        look_back = kpi.params.get('look_back', None)
        box_pts = kpi.params.get('box_pts', None)
        obs_size = kpi.params.get('obs_size', None)
        outlier_ratio_th = kpi.params.get('outlier_ratio_th', None)

        start, end = dt.last(minutes=look_back)
        ts_list = self.data_loader.get_metric(
            start, end, kpi.metric, label_name='machine_id', label_value=machine_id)

        if not ts_list:
            logger.warning(f'Key metric {kpi.metric} is null on the target machine {machine_id}!')
            return []

        point_count = self.data_loader.expected_point_length(start, end)
        anomalies = []
        for _ts in ts_list:
            if len(_ts.values) < point_count * 0.9 or len(_ts.values) > point_count * 1.5:
                continue

            smoothed_val = conv_smooth(_ts.values, box_pts=box_pts)
            outlier, mean, std = three_sigma(smoothed_val, obs_size=obs_size, method='min')
            ratio = divide(len(outlier), obs_size)

            if outlier and ratio >= outlier_ratio_th:
                logger.info(f'Ratio: {ratio}, Outlier Ratio TH: {outlier_ratio_th}, '
                            f'Mean: {mean}, Std: {std}')
                anomalies.append(
                    Anomaly(metric=_ts.metric,
                            labels=_ts.labels,
                            score=ratio,
                            entity_name=kpi.entity_name,
                            description=kpi.description))

        anomalies = sorted(anomalies, key=lambda x: x.score, reverse=True)

        if anomalies:
            logger.info(f'{len(anomalies)} anomalies was detected on sli-rtt model.')

        return anomalies

    def detect_tps(self, kpi, machine_id: str) -> List[Anomaly]:
        """Detects tps by rule based model"""
        look_back = kpi.params.get('look_back', None)
        box_pts = kpi.params.get('box_pts', None)
        obs_size = kpi.params.get('obs_size', None)
        outlier_ratio_th = kpi.params.get('outlier_ratio_th', None)

        start, end = dt.last(minutes=look_back)
        time_series_list = self.data_loader.get_metric(
            start, end, kpi.metric, label_name='machine_id', label_value=machine_id)

        if not time_series_list:
            logger.warning(f'Key metric {kpi.metric} is null on the target machine {machine_id}!')
            return []

        point_count = self.data_loader.expected_point_length(start, end)
        anomalies = []
        for _ts in time_series_list:
            if len(_ts.values) < point_count * 0.9 or len(_ts.values) > point_count * 1.5:
                continue
            smoothed_val = conv_smooth(_ts.values, box_pts=box_pts)
            outlier, mean, std = three_sigma(smoothed_val, obs_size=obs_size, method='min')
            ratio = divide(len(outlier), obs_size)

            if outlier and ratio >= outlier_ratio_th:
                logger.info(f'Ratio: {ratio}, Outlier Ratio TH: {outlier_ratio_th}, '
                            f'Mean: {mean}, Std: {std}')
                anomalies.append(
                    Anomaly(metric=_ts.metric,
                            labels=_ts.labels,
                            score=ratio,
                            entity_name=kpi.entity_name,
                            description=kpi.description))

        anomalies = sorted(anomalies, key=lambda x: x.score, reverse=True)

        if anomalies:
            logger.info(f'{len(anomalies)} anomalies was detected on sli-tps model.')

        return anomalies

    def detect_features(self, metrics, machine_id: str, top_n):
        start, end = dt.last(minutes=6)
        time_series_list = []
        for metric in metrics:
            time_series = self.data_loader.get_metric(
                start, end, metric, label_name='machine_id', label_value=machine_id)
            time_series_list.extend(time_series)

        point_count = self.data_loader.expected_point_length(start, end)
        sr_model = SpectralResidual(12, 24, 50)

        result = []
        for time_series in time_series_list:
            if len(time_series.values) < point_count * 0.9 or \
                    len(time_series.values) > point_count * 1.5:
                continue

            values = time_series.values

            if all(x == values[0] for x in values):
                continue

            scores = sr_model.compute_score(values)
            score = max(scores[-13:])

            if math.isnan(score) or math.isinf(score):
                continue

            result.append((time_series, score))

        result = sorted(result, key=lambda x: x[1], reverse=True)

        return result[0: top_n]

    def report(self, anomaly: Anomaly, machine_id: str):
        """Reports a single anomaly at each time"""
        feature_metrics = [f.metric for f in self.features]
        description = {f.metric: f.description for f in self.features}
        cause_metrics = self.detect_features(feature_metrics, machine_id, top_n=60)
        cause_metrics = [
            {'metric': cause[0].metric,
             'label': cause[0].labels,
             'score': cause[1],
             'description': description.get(cause[0].metric, '')}
            for cause in cause_metrics]
        timestamp = dt.utc_now()
        template = AppAnomalyTemplate(timestamp, machine_id, anomaly.metric, anomaly.entity_name)
        self.anomaly_report.sent_anomaly(anomaly, cause_metrics, template)
