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
from anteater.model.detector.online_vae_detector import OnlineVAEDetector
from anteater.model.detector.tcp_trans_latency_n_sigma_detector import \
    TcpTransLatencyNSigmaDetector
from anteater.module.base import E2EDetector
from anteater.source.anomaly_report import AnomalyReport
from anteater.source.metric_loader import MetricLoader
from anteater.template.sys_anomaly_template import SysAnomalyTemplate


class SysTcpTransmissionLatencyDetector(E2EDetector):
    """SYS tcp transmission latency e2e detector which detects
    the tcp transmission performance deteriorates.
    """

    config_file = 'sys_tcp_transmission_latency.json'

    def __init__(self, data_loader: MetricLoader, reporter: AnomalyReport):
        """The system tcp transmission latency e2e detector initializer"""
        super().__init__(reporter, SysAnomalyTemplate)

        self.detectors = self.init_detectors(data_loader)

    def init_detectors(self, data_loader):
        if self.job_config.model_config.enable:
            detectors = [
                OnlineVAEDetector(data_loader, self.job_config.model_config)
            ]
        else:
            detectors = [
                TcpTransLatencyNSigmaDetector(data_loader)
            ]

        return detectors

    def parse_cause_metrics(self, anomaly: Anomaly) -> List[Dict]:
        """Parses the cause metrics into the specific formats"""
        cause_metrics = [
            {
                'metric': cause.ts.metric,
                'labels': cause.ts.labels,
                'score': cause.score,
                'description': cause.description.format(
                    cause.ts.labels.get('tgid', ''),
                    cause.ts.labels.get('client_port', ''),
                    cause.ts.labels.get('server_ip', ''),
                    cause.ts.labels.get('server_port', ''))}
            for cause in anomaly.root_causes]

        return cause_metrics
