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

from anteater.config import AnteaterConf
from anteater.module.base import E2EDetector
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger


class AnomalyDetection:
    """The anomaly detection base class"""

    def __init__(self, detectors: List[E2EDetector], conf: AnteaterConf):
        """The anomaly detector initializer"""
        self.detectors = detectors
        self.conf = conf

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
