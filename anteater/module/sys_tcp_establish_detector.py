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

from functools import reduce
from typing import List

import numpy as np

from anteater.core.anomaly import Anomaly, CauseMetric
from anteater.module.detector import Detector
from anteater.source.anomaly_report import AnomalyReport
from anteater.source.metric_loader import MetricLoader
from anteater.template.sys_anomaly_template import SysAnomalyTemplate
from anteater.utils.common import divide, same_intersection_key_value
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
        file_name = 'sys_tcp_establish.json'
        super().__init__(data_loader, anomaly_report, file_name)

        self.mean = None
        self.std = None

    def pre_process(self):
        """Calculates ts values mean and std"""
        kpi = self.kpis[0]
        look_back = kpi.params.get('look_back', None)

        start, _ = dt.last(minutes=look_back)
        mid, _ = dt.last(minutes=3)

        ts_list = self.data_loader.get_metric(start, mid, kpi.metric)
        establish_time = reduce(lambda x, y: x + y, [list(set(_ts.values)) for _ts in ts_list])

        self.mean = np.mean(establish_time)
        self.std = np.std(establish_time)

    def execute_detect(self, machine_id: str):
        """Executes the detector based on machine id"""
        kpi = self.kpis[0]
        outlier_ratio_th = kpi.params.get('outlier_ratio_th', None)

        start, end = dt.last(minutes=3)
        ts_list = self.data_loader. \
            get_metric(start, end, kpi.metric, label_name='machine_id', label_value=machine_id)
        establish_time = reduce(lambda x, y: x + y, [list(set(_ts.values)) for _ts in ts_list])

        outlier = [val for val in establish_time if abs(val - self.mean) > 3 * self.std]
        ratio = divide(len(outlier), len(establish_time))
        if outlier and ratio > outlier_ratio_th:
            logger.info(f'Ratio: {ratio}, Outlier Ratio TH: {outlier_ratio_th}, '
                        f'Mean: {self.mean}, Std: {self.std}')
            logger.info('Sys tcp establish anomalies was detected.')
            anomaly = Anomaly(
                metric=kpi.metric,
                labels={},
                entity_name=kpi.entity_name,
                description=kpi.description.format(ratio, min(outlier)))
            self.report(anomaly, machine_id)

    def find_cause_metrics(self, machine_id: str, filters: dict) -> List[CauseMetric]:
        """Detects the abnormal features and reports the caused metrics"""
        priorities = {f.metric: f.priority for f in self.features}
        start, end = dt.last(minutes=3)
        ts_list = []
        for metric in priorities.keys():
            _ts_list = self.data_loader.\
                get_metric(start, end, metric, label_name='machine_id', label_value=machine_id)
            filtered_ts_list = self.filter_ts(_ts_list, filters)
            ts_list.extend(filtered_ts_list)

        result = []
        for _ts in ts_list:
            if _ts.values and max(_ts.values) > 0:
                cause_metric = CauseMetric(ts=_ts, score=max(_ts.values))
                result.append(cause_metric)

        result = sorted(result, key=lambda x: x.score, reverse=True)

        return result

    def report(self, anomaly: Anomaly, machine_id: str):
        """Reports a single anomaly at each time"""
        description = {f.metric: f.description for f in self.features}
        cause_metrics = self.find_cause_metrics(machine_id, anomaly.labels)
        cause_metrics = [
            {
                'metric': cause.ts.metric,
                'label': cause.ts.labels,
                'score': cause.score,
                'description': description.get(cause.ts.metric, '').format(
                    cause.ts.labels.get('ppid', ''),
                    cause.ts.labels.get('s_port', ''))
            }
            for cause in cause_metrics]

        timestamp = dt.utc_now()
        template = SysAnomalyTemplate(timestamp, machine_id, anomaly.metric, anomaly.entity_name)
        self.anomaly_report.sent_anomaly(anomaly, cause_metrics, template)
