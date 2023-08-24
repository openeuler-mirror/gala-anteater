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

from anteater.core.anomaly import Anomaly
from anteater.core.kpi import KPI
from anteater.model.detector.base import Detector
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger


class ThBaseDetector(Detector):
    """The threshold-based anomaly detector"""

    def detect_kpis(self, kpis: List[KPI]):
        """Executes anomaly detection on kpis"""
        start, end = dt.last(minutes=1)
        machine_ids = self.get_unique_machine_id(start, end, kpis)
        anomalies = []
        for _id in machine_ids:
            for kpi in kpis:
                anomalies.extend(self.detect_signal_kpi(kpi, _id))

        return anomalies

    def detect_signal_kpi(self, kpi, machine_id: str) -> List[Anomaly]:
        """Detects kpi based on threshold based anomaly detection model"""
        look_back = kpi.params.get('look_back')
        th = kpi.params.get('th')
        start, end = dt.last(minutes=look_back)
        ts_list = self.data_loader.get_metric(start, end, kpi.metric, machine_id=machine_id)

        if not ts_list:
            logger.warning(f'Key metric {kpi.metric} is null on the target machine {machine_id}!')
            return []

        anomalies = [
            Anomaly(
                machine_id=machine_id,
                metric=_ts.metric,
                labels=_ts.labels,
                score=1,
                entity_name=kpi.entity_name)
            for _ts in ts_list
            if sum(_ts.values) >= th
        ]

        return anomalies
