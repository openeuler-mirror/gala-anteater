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
import os
import sys
import signal
import datetime
import torch
import numpy as np
import random

from apscheduler.schedulers.blocking import BlockingScheduler

from anteater.config import AnteaterConf
from anteater.core.info import MetricInfo
from anteater.provider.kafka import KafkaProvider
from anteater.source.anomaly_report import AnomalyReport
from anteater.source.metric_loader import MetricLoader
from anteater.source.suppress import AnomalySuppression
from anteater.utils.constants import ANTEATER_CONFIG_PATH
from anteater.utils.log import logger
from anteater.anomaly_detection import AnomalyDetection
from anteater.root_cause_analysis import RootCauseAnalysis

PID_FILE = '/var/run/gala-anteater.pid'


def is_running(pid_name):
    try:
        os.kill(pid_name, 0)
    except OSError:
        return False
    else:
        return True


def init_nn_seed(seed_value=110):
    """Make nn methods result can reproduce."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)


def init_config() -> AnteaterConf:
    """initialize anteater config"""
    conf = AnteaterConf()
    conf.load_from_yaml(ANTEATER_CONFIG_PATH)

    return conf


def main():
    """The gala-anteater main function"""
    init_nn_seed()
    conf = init_config()
    kafka_provider = KafkaProvider(conf.kafka)
    metricinfo = MetricInfo()
    suppressor = AnomalySuppression(conf.suppression_time)
    report = AnomalyReport(kafka_provider, suppressor, metricinfo)
    loader = MetricLoader(metricinfo, conf)
    ad = AnomalyDetection(loader, report)
    rca = RootCauseAnalysis(kafka_provider, report, conf.arangodb)

    duration = conf.schedule.duration
    logger.info('Schedule recurrent job, interval %d minute(s).', duration)
    ad.run()
    rca.run()

    scheduler = BlockingScheduler(timezone='Asia/Shanghai')
    scheduler.add_job(ad.run, trigger='interval', minutes=duration, next_run_time=datetime.datetime.now())
    scheduler.add_job(rca.run, trigger='interval', minutes=duration, next_run_time=datetime.datetime.now())
    scheduler.start()


if __name__ == '__main__':
    if os.path.exists(PID_FILE):
        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())
        if is_running(pid):
            logger.info(f"Another instance of Gala-anteater is already running with PID {pid}.")
            sys.exit(1)

    pid = os.getpid()
    f = os.open(PID_FILE, os.O_RDWR | os.O_CREAT, 0o640)
    os.write(f, bytes(str(pid), encoding="utf-8"))
    os.close(f)

    def cleanup(signum, frame):
        os.remove(PID_FILE)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    main()
