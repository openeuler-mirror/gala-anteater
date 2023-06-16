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
import logging
from typing import Dict, List, Type

from anteater.core.anomaly import Anomaly
from anteater.model.detector.base import Detector
from anteater.source.anomaly_report import AnomalyReport
from anteater.source.template import Template
from anteater.utils.data_load import load_job_config


class E2EDetector:
    """The E2E anomaly detection for a specific scenario base class,
    including: data preparation, anomaly detection, adn result reporting.
    """

    config_file = None

    def __init__(self, reporter: AnomalyReport, template: Type[Template]):
        """The E2E anomaly detection base class initializer"""
        self.reporter = reporter
        self.template = template
        self.job_config = load_job_config(self.config_file)

        self.detectors: List[Detector] = []

    def execute(self):
        """Start to run each signal configured detectors"""
        if not self.job_config.kpis:
            logging.info('Empty kpis for detector: %s, skip it!',
                         self.__class__.__name__)
            return

        logging.info('Run E2E detector: %s', self.__class__.__name__)
        for detector in self.detectors:
            anomalies = detector.execute(self.job_config)
            for anomaly in anomalies:
                self.report(anomaly, self.job_config.keywords)

    def report(self, anomaly: Anomaly, keywords):
        """Parses the anomaly into a specific formats
        based on the template and reports parsed results
        """
        template = self.template()
        self.reporter.sent_anomaly(anomaly, keywords, template)
