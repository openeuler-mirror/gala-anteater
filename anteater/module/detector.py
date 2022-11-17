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

from abc import abstractmethod, ABC

from anteater.core.anomaly import Anomaly
from anteater.source.anomaly_report import AnomalyReport
from anteater.source.metric_loader import MetricLoader
from anteater.utils.common import same_intersection_key_value
from anteater.utils.data_load import load_kpi_feature
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger
from anteater.utils.timer import timer


class Detector(ABC):
    """The anomaly detector base class"""
    def __init__(
            self,
            data_loader: MetricLoader,
            anomaly_report: AnomalyReport,
            file_name: str):
        """The detector base class initializer"""
        self.data_loader = data_loader
        self.anomaly_report = anomaly_report
        self.kpis, self.features = load_kpi_feature(file_name)

    @staticmethod
    def filter_ts(ts_list, filters):
        result = []
        for _ts in ts_list:
            if same_intersection_key_value(_ts.labels, filters):
                result.append(_ts)

        return result

    @abstractmethod
    def execute_detect(self, machine_id):
        """Executes anomaly detection on specified machine id"""
        pass

    @abstractmethod
    def report(self, anomaly: Anomaly, machine_id: str):
        """Reports a single anomaly at each time"""
        pass

    @timer
    def detect(self):
        """The main function of detector"""
        if not self.kpis:
            logger.debug(f"Null kpis in detector: {self.__class__.__name__}!")
            return

        logger.info(f"Run detector: {self.__class__.__name__}!")
        self.pre_process()
        machine_ids = self.get_unique_machine_id()
        for _id in machine_ids:
            self.execute_detect(_id)

    def pre_process(self):
        """Executes pre-process for generating necessary parameters"""
        pass

    def get_unique_machine_id(self):
        """Gets unique machine ids during past minutes"""
        start, end = dt.last(minutes=1)
        metrics = [_kpi.metric for _kpi in self.kpis]
        machine_ids = self.data_loader.get_unique_machines(start, end, metrics)
        return machine_ids
