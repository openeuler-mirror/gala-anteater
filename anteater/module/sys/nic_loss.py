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

from anteater.model.detector.th_base_detector import ThBaseDetector
from anteater.module.base import E2EDetector
from anteater.source.anomaly_report import AnomalyReport
from anteater.source.metric_loader import MetricLoader
from anteater.source.template import SysAnomalyTemplate


class NICLossDetector(E2EDetector):
    """SYS nic loss e2e detector which detects the network loss.
    """

    config_file = 'sys_nic_loss.json'

    def __init__(self, data_loader: MetricLoader, reporter: AnomalyReport):
        """The system tcp transmission latency e2e detector initializer"""
        super().__init__(reporter, SysAnomalyTemplate)

        self.detectors = [
            ThBaseDetector(data_loader)
        ]
