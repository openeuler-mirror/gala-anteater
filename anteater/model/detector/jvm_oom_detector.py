#!/usr/bin/python3
# ******************************************************************************
# Copyright (c) 2023 Huawei Technologies Co., Ltd.
# gala-anteater is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# ******************************************************************************/

from collections import defaultdict
from typing import List, Dict

import numpy as np

from anteater.core.anomaly import Anomaly
from anteater.core.kpi import KPI, Feature
from anteater.core.ts import TimeSeries
from anteater.model.detector.base import Detector
from anteater.utils.common import to_bytes
from anteater.utils.constants import AREA, THRESHOLD, TGID,\
    GC, LOOK_BACK, PS_OLD_G, METASPACE, POOL, OLD_G_COLLECTORS,\
    POINTS_MINUTE
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.log import logger
from anteater.utils.timer import timer


def get_kpi(kpis: List[KPI], name: str) -> KPI:
    """Gets the first value matched the name from kpis,
    if not existed, raise exception.
    """
    for item in kpis:
        if name in item.metric:
            return item

    raise ValueError(f'Cannot get kpi matching the name {name}!')


def count_per_minutes(values: List):
    """Calculates incremental count per minutes for the
    time seies values, return empty for null values.
    """
    if len(values) <= POINTS_MINUTE:
        return np.array([])

    for i in range(1, len(values)):
        if values[i-1] > values[i]:
            values[i] = values[i-1]

    return np.subtract(values[POINTS_MINUTE:], values[:-POINTS_MINUTE])


class JVMOOMDetector(Detector):
    """The jvm out-of-memory detector

    This detector mainly leverages the rule-based model as foundation, and
    will predict or detect the potential OOM errors in Java Virtual Machine.
    """

    def detect_out_of_memory(self, kpis, machine_id: str) -> List[Anomaly]:
        """Detects the oom errors based on the model kpis"""
        anomalies = []
        gc_abnormal = self.check_gc(kpis, machine_id)
        for tgid, _ in gc_abnormal.items():
            areas = self.check_memory(kpis, machine_id, tgid)
            if areas:
                anomalies.extend(self.detect_memory_pool(kpis, machine_id, tgid, PS_OLD_G))
                anomalies.extend(self.detect_memory_pool(kpis, machine_id, tgid, METASPACE))

            anomalies.extend(self.detect_jvm_class(kpis, machine_id, tgid))

        anomalies.extend(self.detect_thread(kpis, machine_id))
        anomalies.extend(self.detect_direct_buffer(kpis, machine_id))

        return anomalies

    def detect_kpis(self, kpis: List[KPI]) -> List[Anomaly]:
        """Executes anomaly detection on kpis"""
        return NotImplemented

    def load_time_series(self, kpi, **kwargs) -> List[TimeSeries]:
        """Loads time series of the target kpi"""
        start, end = dt.last(minutes=kpi.params.get(LOOK_BACK))
        metric = kpi.metric
        time_series = self.data_loader.get_metric(start, end, metric, **kwargs)

        return time_series

    def check_gc(self, kpis: List[KPI], machine_id: str) -> Dict[str, List]:
        """Checks the garbage collection status of the JVM"""
        kpi = get_kpi(kpis, 'gc_coll_secs_count')
        time_series = self.load_time_series(kpi, machine_id=machine_id)
        time_series = [ts for ts in time_series if ts.labels.get(GC) in OLD_G_COLLECTORS]
        threshold = kpi.params.get(THRESHOLD)

        gc_abnormal = defaultdict(list)
        for _ts in time_series:
            max_count = np.max(count_per_minutes(_ts.values), initial=0)
            if max_count >= threshold:
                tgid = _ts.labels.get(TGID)
                gc_type = _ts.labels.get(GC)
                gc_abnormal[tgid].append(gc_type)

        return gc_abnormal

    def check_memory(self, kpis: List[KPI], machine_id: str, tgid: str) -> List[str]:
        """Checks the memory status of the JVM"""
        kpi = get_kpi(kpis, 'mem_bytes_max')
        time_series = self.load_time_series(kpi, machine_id=machine_id, tgid=tgid)
        max_values = {_ts.labels.get(AREA): _ts.values[-1] for _ts in time_series}
        max_values = {key: val for key, val in max_values.items() if val != -1}
        if not max_values:
            return []

        areas = []
        kpi = get_kpi(kpis, 'mem_bytes_used')
        time_series = self.load_time_series(kpi, machine_id=machine_id, tgid=tgid)
        time_series = [_ts for _ts in time_series if _ts.labels.get(AREA) in max_values]
        threshold = kpi.params.get(THRESHOLD)
        for _ts in time_series:
            area = _ts.labels.get(AREA)
            usage = np.max(np.divide(_ts.values, max_values.get(area)), initial=0)
            if usage >= threshold:
                areas.append(area)

        return areas

    def detect_memory_pool(self, kpis: List[KPI], machine_id: str, tgid: str, pool: str)\
            -> List[Anomaly]:
        """Detects the memory pools abnormal status of the JVM"""
        kpi = get_kpi(kpis, 'mem_pool_bytes_max')
        time_series = self.load_time_series(kpi, machine_id=machine_id, tgid=tgid, pool=pool)
        max_values = {_ts.labels.get(POOL): _ts.values[-1] for _ts in time_series}
        max_values = {key: val for key, val in max_values.items() if val != -1}
        if not max_values:
            return []

        anomalies = []
        kpi = get_kpi(kpis, 'mem_pool_bytes_used')
        time_series = self.load_time_series(kpi, machine_id=machine_id, tgid=tgid, pool=pool)
        time_series = [_ts for _ts in time_series if _ts.labels.get(POOL) in max_values]
        threshold = kpi.params.get(THRESHOLD)
        for _ts in time_series:
            pool = _ts.labels.get(POOL)
            usage = np.max(np.divide(_ts.values, max_values.get(pool)), initial=0)
            if usage >= threshold:
                anomalies.append(Anomaly(
                    machine_id=machine_id,
                    metric=_ts.metric,
                    labels=_ts.labels,
                    entity_name=kpi.entity_name,
                    details={f'\'{pool}\'Usage': usage},
                    score=-1))

        return anomalies

    def detect_jvm_class(self, kpis: List[KPI], machine_id: str, tgid: str) -> List[Anomaly]:
        """Detects classes abnormal status of the JVM"""
        kpi = get_kpi(kpis, 'class_current_loaded')
        time_series = self.load_time_series(kpi, machine_id=machine_id, tgid=tgid)
        threshold = kpi.params[THRESHOLD]
        anomalies = []
        for _ts in time_series:
            count = (np.average(_ts.values) if _ts.values else 0)
            if count >= threshold:
                anomalies.append(Anomaly(
                    machine_id=machine_id,
                    metric=_ts.metric,
                    labels=_ts.labels,
                    entity_name=kpi.entity_name,
                    details={'LoadedClassCount': round(count)},
                    score=-1))

        return anomalies

    def detect_thread(self, kpis: List[KPI], machine_id: str) -> List[Anomaly]:
        """Detects threads abnormal status of the JVM"""
        kpi = get_kpi(kpis, 'threads_current')
        time_series = self.load_time_series(kpi, machine_id=machine_id)
        threshold = kpi.params.get(THRESHOLD)
        anomalies = []
        for _ts in time_series:
            count = (np.average(_ts.values) if _ts.values else 0)
            if count >= threshold:
                anomalies.append(Anomaly(
                    machine_id=machine_id,
                    metric=_ts.metric,
                    labels=_ts.labels,
                    entity_name=kpi.entity_name,
                    details={'ThreadsCount': round(count)},
                    score=-1))

        return anomalies

    def detect_direct_buffer(self, kpis: List[KPI], machine_id: str) -> List[Anomaly]:
        """Detects direct buffer abnormal status of the JVM"""
        kpi = get_kpi(kpis, 'buffer_pool_used_bytes')
        time_series = self.load_time_series(kpi, machine_id=machine_id)
        threshold = kpi.params.get(THRESHOLD)
        threshold = to_bytes(threshold)
        anomalies = []
        for _ts in time_series:
            count = (np.average(_ts.values) if _ts.values else 0)
            if count >= threshold:
                anomalies.append(Anomaly(
                    machine_id=machine_id,
                    metric=_ts.metric,
                    labels=_ts.labels,
                    entity_name=kpi.entity_name,
                    details={'DirectBufferMem': count},
                    score=-1))

        return anomalies

    @timer
    def _execute(self, kpis: List[KPI], features: List[Feature], **kwargs) -> List[Anomaly]:
        """Executes the oom anomaly detection algorithms"""
        jvm_info = get_kpi(kpis, 'jvm_info')
        start, end = dt.last(minutes=jvm_info.params.get(LOOK_BACK))
        machine_ids = self.get_unique_machine_id(start, end, [jvm_info])

        anomalies = []
        for _id in machine_ids:
            anomalies.extend(self.detect_out_of_memory(kpis, _id))

        if anomalies:
            logger.info(f'{len(anomalies)} anomalies was detected on {self.__class__.__name__}.')

        return anomalies
