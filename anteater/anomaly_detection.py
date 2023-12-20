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

from typing import List

from anteater.module.base import E2EDetector
from anteater.source.anomaly_report import AnomalyReport
from anteater.source.metric_loader import MetricLoader
from anteater.utils.data_load import load_jobs
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger


class AnomalyDetection:
    """The anomaly detection base class"""

    def __init__(self, loader: MetricLoader, reporter: AnomalyReport):
        """The anomaly detector initializer"""
        self.loader = loader
        self.reporter = reporter

        self.detectors = self.load_detectors()

    def load_detectors(self) -> List[E2EDetector]:
        """load detectors from config file"""
        detectors = []
        stop_count = 0
        for job_config in load_jobs():
            if job_config.enable and job_config.job_type == 'anomaly_detection':
                detectors.append(E2EDetector(self.loader, self.reporter, job_config))
            else:
                stop_count += 1

        logger.info('Anomaly detection loaded %d jobs.', len(detectors))
        logger.debug('There are %d jobs was stopped!', stop_count)

        return detectors

    def append(self, detector: E2EDetector):
        """Appends the detector"""
        self.detectors.append(detector)

    def run(self):
        """Executes the detectors for anomaly detection"""
        dt.update_and_freeze()
        logger.info('START: A new gala-anteater task is scheduling!')
        for detector in self.detectors:
            detector.execute()
        logger.info('END!')
