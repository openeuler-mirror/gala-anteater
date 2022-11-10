#!/usr/bin/python3
# ******************************************************************************
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
# licensed under the Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#     http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
# PURPOSE.
# See the Mulan PSL v2 for more details.
# ******************************************************************************/
"""
Time:
Author:
Description: The main function of gala-anteater project.
"""

from functools import partial

from apscheduler.schedulers.blocking import BlockingScheduler

from anteater.config import AnteaterConf
from anteater.module.app_sli_detector import APPSliDetector
from anteater.module.proc_io_latency_detector import ProcIOLatencyDetector
from anteater.module.sys_io_latency_detector import SysIOLatencyDetector
from anteater.module.sys_tcp_establish_detector import SysTcpEstablishDetector
from anteater.module.sys_tcp_transmission_detector import SysTcpTransmissionDetector
from anteater.provider.kafka import KafkaProvider
from anteater.source.anomaly_report import AnomalyReport
from anteater.source.metric_loader import MetricLoader
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger

ANTEATER_DATA_PATH = '/etc/gala-anteater/'


def init_config() -> AnteaterConf:
    """initialize anteater config"""
    conf = AnteaterConf()
    conf.load_from_yaml(ANTEATER_DATA_PATH)

    return conf


def anomaly_detection(loader: MetricLoader, report: AnomalyReport, conf: AnteaterConf):
    """Run anomaly detection model periodically"""
    dt.update_and_freeze()
    logger.info('START: anomaly detection!')

    # APP sli anomaly detection
    APPSliDetector(loader, report).detect()

    # SYS tcp/io detection
    SysTcpEstablishDetector(loader, report).detect()
    SysTcpTransmissionDetector(loader, report).detect()
    SysIOLatencyDetector(loader, report).detect()
    ProcIOLatencyDetector(loader, report).detect()

    logger.info('END: anomaly detection!')


def main():
    conf = init_config()

    kafka_provider = KafkaProvider(conf.kafka)
    loader = MetricLoader(conf)
    report = AnomalyReport(kafka_provider)

    logger.info(f'Schedule recurrent job with time interval {conf.schedule.duration} minute(s).')

    scheduler = BlockingScheduler(timezone='Asia/Shanghai')
    scheduler.add_job(partial(anomaly_detection, loader, report, conf),
                      trigger='interval', minutes=conf.schedule.duration)
    scheduler.start()


if __name__ == '__main__':
    main()
