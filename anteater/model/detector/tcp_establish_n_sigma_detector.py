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

from functools import reduce
from typing import List

import numpy as np

from anteater.core.anomaly import Anomaly
from anteater.core.kpi import KPI
from anteater.model.detector.base import Detector
from anteater.source.metric_loader import MetricLoader
from anteater.utils.common import divide
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger


class TcpEstablishNSigmaDetector(Detector):
    """The anomaly detector base class"""

    def __init__(self, data_loader: MetricLoader):
        """The detector base class initializer"""
        super().__init__(data_loader)

        self.mean = None
        self.std = None

    def update_global_mean_std(self, kpi):
        """Calculates ts values mean and std"""
        look_back = kpi.params.get('look_back', None)

        start, _ = dt.last(minutes=look_back)
        mid, _ = dt.last(minutes=3)

        ts_list = self.data_loader.get_metric(start, mid, kpi.metric)
        establish_time = reduce(lambda x, y: x + y, [list(set(_ts.values)) for _ts in ts_list])

        self.mean = np.mean(establish_time)
        self.std = np.std(establish_time)

    def detect_kpis(self, kpis: List[KPI]) -> List[Anomaly]:
        """Executes anomaly detection on kpis"""
        self.update_global_mean_std(kpis[0])

        start, end = dt.last(minutes=1)
        machine_ids = self.get_unique_machine_id(start, end, kpis)
        anomalies = []
        for _id in machine_ids:
            for kpi in kpis:
                anomalies.extend(self.detect_signal_kpi(kpi, _id))

        return anomalies

    def detect_signal_kpi(self, kpi, machine_id: str) -> List[Anomaly]:
        """Detects kpi based on signal time series anomaly detection model"""
        outlier_ratio_th = kpi.params.get('outlier_ratio_th')
        look_back = kpi.params.get('obs_size')

        start, end = dt.last(minutes=look_back)
        ts_list = self.data_loader.\
            get_metric(start, end, kpi.metric, label_name='machine_id', label_value=machine_id)

        anomalies = []
        for _ts in ts_list:
            outlier = [val for val in _ts.values if abs(val - self.mean) > 3 * self.std]
            ratio = divide(len(outlier), len(_ts.values))
            if outlier and ratio > outlier_ratio_th:
                anomalies.append(
                    Anomaly(
                        machine_id=machine_id,
                        metric=kpi.metric,
                        labels=_ts.labels,
                        score=ratio,
                        entity_name=kpi.entity_name,
                        description=kpi.description.format(max(_ts.values)))
                )

        if anomalies:
            logger.info(f'{len(anomalies)} anomalies was detected on {self.__class__.__name__}.')

        return anomalies
