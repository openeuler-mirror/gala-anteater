#!/usr/bin/python3
# ******************************************************************************
# Copyright (c) 2023 Huawei Technologies Co., Ltd.
# gala-anteater is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# ******************************************************************************/

from anteater.model.detector.jvm_oom_detector import JVMOOMDetector
from anteater.module.base import E2EDetector
from anteater.source.anomaly_report import AnomalyReport
from anteater.source.metric_loader import MetricLoader
from anteater.source.template import JVMAnomalyTemplate


class JVMOutOfMemoryDetector(E2EDetector):
    """JVM OutOfMemory detector which detects some common OutOfMemory
    errors inside the java virtual machine, including Heapspace, GC,
    Metaspace, etc.
    """

    config_file = 'jvm_oom.json'

    def __init__(self, data_loader: MetricLoader, reporter: AnomalyReport):
        """The JVM OutOfMemory E2E detector initializer"""

        super().__init__(reporter, JVMAnomalyTemplate)

        self.detectors = [
            JVMOOMDetector(data_loader)
        ]
