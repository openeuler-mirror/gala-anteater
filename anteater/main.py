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
Description: The main function of gala-anteater project.
"""

from apscheduler.schedulers.blocking import BlockingScheduler

from anteater.anomaly_detection import AnomalyDetection
from anteater.config import AnteaterConf
from anteater.module.app.app_sli_detector import APPSliDetector
from anteater.module.jvm.jvm_oom_detector import JVMOutOfMemoryDetector
from anteater.module.sys.disk_throughput import DiskThroughputDetector
from anteater.module.sys.nic_loss import NICLossDetector
from anteater.module.sys.proc_io_latency import ProcIOLatencyDetector
from anteater.module.sys.sys_io_latency import SysIOLatencyDetector
from anteater.module.sys.tcp_establish import SysTcpEstablishDetector
from anteater.module.sys.tcp_transmission_latency import SysTcpTransmissionLatencyDetector
from anteater.provider.kafka import KafkaProvider
from anteater.source.anomaly_report import AnomalyReport
from anteater.source.metric_loader import MetricLoader
from anteater.utils.log import logger

ANTEATER_DATA_PATH = '/etc/gala-anteater/'


def init_config() -> AnteaterConf:
    """initialize anteater config"""
    conf = AnteaterConf()
    conf.load_from_yaml(ANTEATER_DATA_PATH)

    return conf


def main():
    conf = init_config()
    kafka_provider = KafkaProvider(conf.kafka)
    loader = MetricLoader(conf)
    report = AnomalyReport(kafka_provider)
    detectors = [
        # APP sli anomaly detection
        APPSliDetector(loader, report),

        # SYS tcp/io detection
        SysTcpEstablishDetector(loader, report),
        SysTcpTransmissionLatencyDetector(loader, report),
        SysIOLatencyDetector(loader, report),
        ProcIOLatencyDetector(loader, report),
        DiskThroughputDetector(loader, report),
        NICLossDetector(loader, report),

        # JVM anomaly detection
        JVMOutOfMemoryDetector(loader, report)
    ]

    anomaly_detect = AnomalyDetection(detectors, conf)

    logger.info(f'Schedule recurrent job with time interval {conf.schedule.duration} minute(s).')
    scheduler = BlockingScheduler(timezone='Asia/Shanghai')
    scheduler.add_job(anomaly_detect.run, trigger='interval', minutes=conf.schedule.duration)
    scheduler.start()


if __name__ == '__main__':
    main()
