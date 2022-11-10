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

import numpy as np

from anteater.core.anomaly import Anomaly
from anteater.module.detector import Detector
from anteater.source.anomaly_report import AnomalyReport
from anteater.source.metric_loader import MetricLoader
from anteater.template.sys_anomaly_template import SysAnomalyTemplate
from anteater.utils.common import divide
from anteater.utils.data_load import load_kpi_feature
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger


class SysTcpEstablishDetector(Detector):
    """SYS tcp establish detector which detects the tcp established
    performance deteriorates. Currently, for the tcp establish kpi,
    we could leverage '3-sigma rule', 'IQR outlier test' or 'Box plots'
    for anomaly detecting.

    - referring: https://stackoverflow.com/questions/2303510/
                 recommended-anomaly-detection-technique-for
                 -simple-one-dimensional-scenario
    """

    def __init__(self, data_loader: MetricLoader, anomaly_report: AnomalyReport):
        super().__init__(data_loader, anomaly_report)
        self.kpis, self.features = load_kpi_feature('sys_tcp_establish.json')

    def execute_detect(self, machine_id: str):
        kpi = self.kpis[0]
        start_30_minutes, _ = dt.last(minutes=30)
        start_3_minutes, end = dt.last(minutes=3)

        pre_ts = self.data_loader.get_metric(
            start_30_minutes, start_3_minutes, kpi.metric, label_name='machine_id', label_value=machine_id)
        pre_establish_time = [t.values[0] for t in pre_ts if t.values]

        ts = self.data_loader.get_metric(
            start_3_minutes, end, kpi.metric, label_name='machine_id', label_value=machine_id)
        establish_time = [t.values[0] for t in ts if t.values]

        mean = np.mean(pre_establish_time)
        std = np.std(pre_establish_time)

        outlier = [val for val in establish_time if abs(val - mean) > 3 * std]

        if outlier and len(outlier) > len(ts) * 0.3:
            logger.info('Sys tcp establish anomalies was detected.')
            if establish_time:
                percentile = divide(len(outlier), len(establish_time))
            else:
                percentile = 0
            anomaly = Anomaly(
                metric=kpi.metric,
                labels={},
                description=kpi.description.format(percentile, min(outlier)))
            self.report(anomaly, kpi.entity_name, machine_id)

    def detect_features(self, machine_id: str):
        start, end = dt.last(minutes=3)
        time_series_list = []
        metrics = [f.metric for f in self.features]
        for metric in metrics:
            time_series = self.data_loader.get_metric(
                start, end, metric, label_name='machine_id', label_value=machine_id)
            time_series_list.extend(time_series)

        result = []
        for ts in time_series_list:
            if ts.values and max(ts.values) > 0:
                result.append((ts, max(ts.values)))

        result = sorted(result, key=lambda x: x[1], reverse=True)

        return result

    def report(self, anomaly: Anomaly, entity_name: str, machine_id: str):
        description = {f.metric: f.description for f in self.features}
        cause_metrics = self.detect_features(machine_id)
        cause_metrics = [
            {'metric': cause[0].metric,
             'label': cause[0].labels,
             'score': cause[1],
             'description': description.get(cause[0].metric, '').format(
                 cause[0].labels.get('ppid', ''),
                 cause[0].labels.get('s_port', ''))}
            for cause in cause_metrics]
        timestamp = dt.utc_now()
        template = SysAnomalyTemplate(timestamp, machine_id, anomaly.metric, entity_name)
        template.labels = anomaly.labels
        self.anomaly_report.sent_anomaly(anomaly, cause_metrics, template)
