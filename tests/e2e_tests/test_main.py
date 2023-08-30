import unittest
from datetime import datetime, timedelta

from dateutil.tz import tzlocal

from anteater.anomaly_detection import AnomalyDetection
from anteater.config import AnteaterConf
from anteater.core.info import MetricInfo
from anteater.source.anomaly_report import AnomalyReport
from anteater.source.suppress import AnomalySuppression
from anteater.utils.datetime import DateTime
from anteater.utils.log import logger
from tests.e2e_tests.test_evalution import TestEvaluation, load_labels
from tests.e2e_tests.test_kafka import TestKafkaProvider
from tests.e2e_tests.test_metric import TestMetricLoader

ANTEATER_DATA_PATH = '/etc/gala-anteater/'


class TestE2E(unittest.TestCase):

    def test_main(self):
        conf = AnteaterConf()
        conf.load_from_yaml(ANTEATER_DATA_PATH)
        kafka_provider = TestKafkaProvider(conf.kafka)
        metricinfo = MetricInfo()
        suppressor = AnomalySuppression()
        loader = TestMetricLoader(conf)
        report = AnomalyReport(kafka_provider, suppressor, metricinfo)
        anomaly_detect = AnomalyDetection(loader, report)

        time_start = datetime.strptime("2023-2-10 08:00:00", "%Y-%m-%d %H:%M:%S").astimezone(tz=tzlocal())
        time_end = datetime.strptime("2023-2-10 22:00:00", "%Y-%m-%d %H:%M:%S").astimezone(tz=tzlocal())

        time_curr = time_start
        while time_curr < time_end:
            DateTime().update(time=time_curr)
            logger.info(f'Run anomaly detection in timestamp {time_curr}')
            anomaly_detect.run()
            time_curr = time_curr + timedelta(minutes=1)

        topic = conf.kafka.model_topic
        anomalies = kafka_provider.received_messages.get(topic)
        labels = load_labels()

        logger.info(f'Total anomalies: {len(anomalies)}')
        label_count = {key: len(labels.get(key)) for key in labels.keys()}
        logger.info(f'Total labels: {label_count}')

        evaluation = TestEvaluation(anomalies, labels)
        evaluation.evaluate(time_start, time_end)
