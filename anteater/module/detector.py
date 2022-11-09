from abc import abstractmethod

from anteater.core.anomaly import Anomaly
from anteater.source.anomaly_report import AnomalyReport
from anteater.source.metric_loader import MetricLoader
from anteater.utils.datetime import datetime_manager
from anteater.utils.log import logger
from anteater.utils.timer import timer


class Detector:
    """The base detector class"""
    def __init__(self, data_loader: MetricLoader, anomaly_report: AnomalyReport):
        self.data_loader = data_loader
        self.anomaly_report = anomaly_report

    @abstractmethod
    def execute_detect(self, machine_id):
        pass

    @timer
    def detect(self):
        logger.info(f"Run detector: {self.__class__.__name__}!")
        start, end = datetime_manager.last(minutes=1)
        metrics_kpi = [k.metric for k in self.kpis]
        metrics_feat = [f.metric for f in self.features]
        metrics = metrics_kpi + metrics_feat
        machine_ids = self.data_loader.get_unique_machines(start, end, metrics)
        for _id in machine_ids:
            self.execute_detect(_id)

    @abstractmethod
    def report(self, anomaly: Anomaly, entity_name: str, machine_id: str):
        pass
