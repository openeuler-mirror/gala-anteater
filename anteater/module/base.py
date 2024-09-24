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

from anteater.core.anomaly import Anomaly
from anteater.core.kpi import JobConfig
from anteater.factory.detectors import DetectorFactory
from anteater.factory.templates import TemplateFactory
from anteater.model.detector.base import Detector
from anteater.source.anomaly_report import AnomalyReport
from anteater.source.metric_loader import MetricLoader
from anteater.source.template import Template
from anteater.utils.log import logger


class E2EDetector:
    """The E2E anomaly detection for a specific scenario base class,
    including: data preparation, anomaly detection, and result reporting.
    """

    config_file = None

    def __init__(self, data_loader: MetricLoader,
                 reporter: AnomalyReport, job_config: JobConfig):
        """The E2E anomaly detection base class initializer"""
        self.data_loader = data_loader
        self.reporter = reporter
        self.job_config = job_config

        self.detector = self.get_detector(
            self.job_config.detector, config=self.job_config.model_config)

        self.template = self.get_template(self.job_config.template)

    def execute(self):
        """Start to run each signal configured detectors"""
        if not self.job_config.kpis:
            logger.info('Empty kpis for detector: %s, skip it!',
                        self.job_config.name)
            return

        logger.info('Run E2E detector: %s', self.job_config.name)
        anomalies = self.detector.execute(self.job_config)
        for anomaly in anomalies:
            self.report(anomaly, self.job_config.keywords)

    def report(self, anomaly: Anomaly, keywords):
        """Parses the anomaly into a specific formats
        based on the template and reports parsed results
        """
        self.reporter.sent_anomaly(anomaly, keywords, self.template)

    def get_detector(self, name, **kwargs) -> Detector:
        """Gets the detector by the name"""
        return DetectorFactory.get_detector(name, self.data_loader, **kwargs)

    def get_template(self, name, **kwargs) -> Template:
        """Gets the template by the name"""
        return TemplateFactory.get_template(name, **kwargs)
