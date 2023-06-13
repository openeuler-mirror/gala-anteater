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
from anteater.module.app.app_sli_detector import APPSliDetector
from anteater.module.base import E2EDetector
from anteater.module.jvm.jvm_oom_detector import JVMOutOfMemoryDetector
from anteater.module.sys.disk_throughput import DiskThroughputDetector
from anteater.module.sys.nic_loss import NICLossDetector
from anteater.module.sys.proc_io_latency import ProcIOLatencyDetector
from anteater.module.sys.sys_io_latency import SysIOLatencyDetector
from anteater.module.sys.tcp_establish import SysTcpEstablishDetector
from anteater.module.sys.tcp_transmission_latency import SysTcpTransmissionLatencyDetector
from anteater.source.anomaly_report import AnomalyReport
from anteater.source.metric_loader import MetricLoader
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger


DetectorCls = [
    # APP sli anomaly detection
    APPSliDetector,

    # SYS tcp/io detection
    SysTcpEstablishDetector,
    SysTcpTransmissionLatencyDetector,
    SysIOLatencyDetector,
    ProcIOLatencyDetector,
    DiskThroughputDetector,
    NICLossDetector,

    # JVM anomaly detection
    JVMOutOfMemoryDetector
]


class AnomalyDetection:
    """The anomaly detection base class"""

    def __init__(self, loader: MetricLoader, reporter: AnomalyReport, conf: AnteaterConf):
        """The anomaly detector initializer"""
        self.loader = loader
        self.reporter = reporter
        self.conf = conf

        self.detectors = self.init_detectors()

    def init_detectors(self) -> List[E2EDetector]:
        """Initializes each detectors"""
        detectors = []
        for cls in DetectorCls:
            detectors.append(cls(self.loader, self.reporter))

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
