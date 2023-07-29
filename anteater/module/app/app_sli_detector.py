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

from anteater.model.detector.usad_detector import UsadDetector
from anteater.module.base import E2EDetector
from anteater.model.detector.online_vae_detector import OnlineVAEDetector
from anteater.model.detector.n_sigma_detector import NSigmaDetector
from anteater.source.anomaly_report import AnomalyReport
from anteater.source.metric_loader import MetricLoader
from anteater.source.template import AppAnomalyTemplate


class APPSliDetector(E2EDetector):
    """APP sli detector which detects the abnormal
    on the sli metrics including rtt, tps, etc.
    """

    # config_file = 'app_sli_rtt.json'
    config_file = 'usad_model.json'

    def __init__(self, data_loader: MetricLoader, reporter: AnomalyReport):
        """The app sli E2E detector initializer"""

        super().__init__(reporter, AppAnomalyTemplate)

        self.detectors = self.init_detectors(data_loader)

    def init_detectors(self, data_loader):
        if self.job_config.model_config.enable:
            detectors = [
                # OnlineVAEDetector(data_loader, self.job_config.model_config)
                UsadDetector(data_loader, self.job_config.model_config)
            ]
        else:
            detectors = [
                NSigmaDetector(data_loader)
            ]

        return detectors
