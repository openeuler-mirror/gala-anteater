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
Time: 2023-06-12
Author: Zhenxing
Description: The main function of gala-anteater project.
"""

from apscheduler.schedulers.blocking import BlockingScheduler

from anteater.anomaly_detection import AnomalyDetection
from anteater.config import AnteaterConf
from anteater.core.info import MetricInfo
from anteater.provider.kafka import KafkaProvider
from anteater.source.anomaly_report import AnomalyReport
from anteater.source.metric_loader import MetricLoader
from anteater.source.suppress import AnomalySuppression
from anteater.utils.constants import ANTEATER_CONFIG_PATH
from anteater.utils.log import logger


def init_config() -> AnteaterConf:
    """initialize anteater config"""
    conf = AnteaterConf()
    conf.load_from_yaml(ANTEATER_CONFIG_PATH)

    return conf


def main():
    """The gala-anteater main function"""
    conf = init_config()
    kafka_provider = KafkaProvider(conf.kafka)
    metricinfo = MetricInfo()
    suppressor = AnomalySuppression()
    report = AnomalyReport(kafka_provider, suppressor, metricinfo)
    loader = MetricLoader(metricinfo, conf)
    anomaly_detection = AnomalyDetection(loader, report)

    duration = conf.schedule.duration
    logger.info('Schedule recurrent job, interval %d minute(s).', duration)

    scheduler = BlockingScheduler(timezone='Asia/Shanghai')
    scheduler.add_job(anomaly_detection.run, trigger='interval', minutes=duration)
    scheduler.start()


if __name__ == '__main__':
    main()
