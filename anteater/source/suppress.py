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

from datetime import datetime, timedelta
from typing import List, Tuple, Dict

from anteater.core.anomaly import Anomaly
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger


class AnomalySuppression:
    """The reported anomaly events suppression

    The same type of anomaly events will be reported multiple times,
    when an abnormal system status sustained a long time period.
    The AnomalySuppression aims to reduce the num of identical anomaly events.
    """

    def __init__(self) -> None:
        """The Anomaly Suppression class initializer"""
        self.look_back = 5    # time to backtrace (minutes)
        self.max_len = 10000  # max length of the queue
        self.ab_queue: List[Tuple[datetime, Anomaly]] = []

    @property
    def ab_machine_ids(self) -> List[str]:
        """Gets recent anomalies' machine ids"""
        return [a.machine_id for t, a in self.ab_queue]

    @property
    def ab_metrics(self) -> List[str]:
        """Gets recent anomalies metrics"""
        return [a.metric for t, a in self.ab_queue]

    @property
    def ab_labels(self) -> List[Dict]:
        """Gets recent anomalies labels"""
        return [a.labels for t, a in self.ab_queue]

    def suppress(self, anomaly: Anomaly) -> bool:
        """Suppresses reported anomalies if there is a same one recently"""
        if self._check_same_type(anomaly):
            logger.info('An anomaly was be supressed by AnomalySuppression!')
            return True

        self._append(anomaly)
        return False

    def _update(self) -> None:
        """Updates recent anomalies"""
        if not self.ab_queue:
            return

        timestamp = dt.utc_now()
        tmp_ts = timestamp - timedelta(minutes=self.look_back)
        self.ab_queue = [x for x in self.ab_queue if x[0] >= tmp_ts]

    def _append(self, anomaly: Anomaly) -> None:
        """Appends anomaly to the queue"""
        if len(self.ab_labels) > self.max_len:
            self.ab_labels.pop(0)

        timestamp = dt.utc_now()
        self.ab_queue.append((timestamp, anomaly))

    def _check_same_type(self, anomaly: Anomaly) -> bool:
        """Checks there are the same machine id, metric, labels
        and descriptions in historical anomalies
        """
        if not anomaly:
            return False

        self._update()
        machine_id = anomaly.machine_id
        metric = anomaly.metric
        labels = anomaly.labels
        details = anomaly.details

        filtered_queue = []
        for _, x in self.ab_queue:
            if machine_id == x.machine_id and \
               metric == x.metric and \
               labels == x.labels and \
               details == x.details:
                filtered_queue.append(x)

        if filtered_queue:
            return True

        return False
