import unittest

import torch
import numpy as np
import random
import time

from datetime import datetime, timedelta

from anteater.config import AnteaterConf
from anteater.core.info import MetricInfo
from anteater.provider.kafka import KafkaProvider
from anteater.source.anomaly_report import AnomalyReport
from anteater.source.metric_loader import MetricLoader
from anteater.source.suppress import AnomalySuppression
from anteater.utils.constants import ANTEATER_CONFIG_PATH
from anteater.utils.log import logger
from anteater.utils.common import GlobalVariable
from anteater.anomaly_detection import AnomalyDetection


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


def disruption_main():
    """The gala-anteater main function"""
    init_nn_seed()
    conf = init_config()
    kafka_provider = KafkaProvider(conf.kafka)
    metricinfo = MetricInfo()
    suppressor = AnomalySuppression()
    report = AnomalyReport(kafka_provider, suppressor, metricinfo)
    loader = MetricLoader(metricinfo, conf)
    ad = AnomalyDetection(loader, report)

    time_start = datetime.strptime("2024-06-24T17:56:00Z", "%Y-%m-%dT%H:%M:%SZ")
    time_end = datetime.strptime("2024-06-24T18:30:00Z", "%Y-%m-%dT%H:%M:%SZ")

    GlobalVariable.is_test_model = True

    time_curr = time_start + timedelta(minutes=15)
    while time_curr < time_end:
        logger.info(f'Run anomaly detection in timestamp {time_curr}')
        GlobalVariable.start_time = time_curr - timedelta(minutes=15)
        GlobalVariable.end_time = time_curr

        ad.run()

        time_curr += timedelta(minutes=1)
        time.sleep(60)



class MyTestCase(unittest.TestCase):
    def test_disruption_detector(self):
        disruption_main()

        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
