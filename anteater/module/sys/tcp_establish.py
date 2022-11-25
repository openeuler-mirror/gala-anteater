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

from typing import List, Dict

from anteater.core.anomaly import Anomaly
from anteater.module.base import E2EDetector
from anteater.model.detector.tcp_establish_n_sigma_detector import\
    TcpEstablishNSigmaDetector
from anteater.source.anomaly_report import AnomalyReport
from anteater.source.metric_loader import MetricLoader
from anteater.template.sys_anomaly_template import SysAnomalyTemplate


class SysTcpEstablishDetector(E2EDetector):
    """SYS tcp establish e2e detector which detects the tcp established
    performance deteriorates. Currently, for the tcp establish kpi,
    we could leverage '3-sigma rule', 'IQR outlier test' or 'Box plots'
    for anomaly detecting.

    - referring: https://stackoverflow.com/questions/2303510/
                 recommended-anomaly-detection-technique-for
                 -simple-one-dimensional-scenario
    """

    config_file = 'sys_tcp_establish.json'

    def __init__(self, data_loader: MetricLoader, reporter: AnomalyReport):
        """The system tcp establish e2e detector initializer"""
        super().__init__(reporter, SysAnomalyTemplate)

        self.detectors = [
            TcpEstablishNSigmaDetector(data_loader),
        ]

    def parse_cause_metrics(self, anomaly: Anomaly) -> List[Dict]:
        """Parses the cause metrics into the specific formats"""
        cause_metrics = [
            {
                'metric': cause.ts.metric,
                'label': cause.ts.labels,
                'score': cause.score,
                'description': cause.description.format(
                    cause.ts.labels.get('ppid', ''),
                    cause.ts.labels.get('s_port', ''))}
            for cause in anomaly.root_causes]

        return cause_metrics


