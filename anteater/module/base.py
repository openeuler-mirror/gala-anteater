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

from abc import abstractmethod
from typing import Callable, Dict, List

from anteater.core.anomaly import Anomaly
from anteater.model.detector.base import Detector
from anteater.source.anomaly_report import AnomalyReport
from anteater.utils.data_load import load_job_config
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger


class E2EDetector:
    """The E2E anomaly detection for a specific scenario base class,
    including: data preparation, anomaly detection, adn result reporting.
    """

    config_file = None

    def __init__(self, reporter: AnomalyReport, template: Callable):
        """The E2E anomaly detection base class initializer"""
        self.reporter = reporter
        self.template = template
        self.job_config = load_job_config(self.config_file)

        self.detectors: List[Detector] = []

    def execute(self):
        """Start to run each signal configured detectors"""
        if not self.job_config.kpis:
            logger.info(f"Empty kpis for detector: {self.__class__.__name__}, skip it!")
            return

        logger.info(f'Run E2E detector: {self.__class__.__name__}')
        for detector in self.detectors:
            anomalies = detector.execute(self.job_config)
            for anomaly in anomalies:
                self.report(anomaly)

    @abstractmethod
    def parse_cause_metrics(self, anomaly: Anomaly) -> List[Dict]:
        """Parses the cause metrics into the specific formats"""
        pass

    def report(self, anomaly: Anomaly):
        """Parses the anomaly into a specific formats
        based on the template and reports parsed results
        """
        cause_metrics = self.parse_cause_metrics(anomaly)
        timestamp = dt.utc_now()
        template = self.template(timestamp, anomaly.machine_id,
                                 anomaly.metric, anomaly.entity_name)
        self.reporter.sent_anomaly(anomaly, cause_metrics, template)
