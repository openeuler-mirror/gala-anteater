#!/usr/bin/python3
# ******************************************************************************
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
# licensed under the Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#     http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
# PURPOSE.
# See the Mulan PSL v2 for more details.
# ******************************************************************************/
"""
Time:
Author:
Description: The key metric model which aims to do the anomaly detection for key metrics.
"""

import math
from datetime import datetime, timedelta
from typing import List, Tuple, Any

import numpy as np

from anteater.config import AnteaterConfig
from anteater.source.metric_loader import MetricLoader
from anteater.utils.data_smooth import smooth
from anteater.utils.log import Log

log = Log().get_logger()


class SLIModel:
    """The SLI (key metric) model which will detect key metric is anomaly or not"""

    def __init__(self, config: AnteaterConfig):
        """The post model initializer"""
        self.config = config
        self.rtt = config.sli_model.rtt
        self.tps = config.sli_model.tps

    def detect(self, utc_now: datetime, machine_id: str):
        """Predicts anomalous score for the time series values"""

        sli_anomalies, default_anomalies = self.detect_rtt(utc_now, machine_id)
        tps_anomalies = self.detect_tps(utc_now, machine_id)

        return sli_anomalies + tps_anomalies, default_anomalies

    def detect_rtt(self, utc_now: datetime, machine_id: str)\
            -> Tuple[List[Tuple[Any, dict, Any]], List[Tuple[Any, dict, Any]]]:
        """Detects rtt by rule-based model"""
        tim_start = utc_now - timedelta(minutes=1)
        tim_end = utc_now

        metric_name = self.rtt.name
        threshold = self.rtt.threshold

        metric_loader = MetricLoader(tim_start, tim_end, self.config)
        labels, values = metric_loader.get_metric(metric_name, label_name="machine_id", label_value=machine_id)

        if not labels or not values:
            log.error(f"Key metric {metric_name} is null on the target machine {machine_id}!")

        scores = []
        for lbl, val in zip(labels, values):
            score = np.mean([np.float64(v[1]) for v in val])
            scores.append((lbl["__name__"], lbl, score))

        sorted_scores = sorted(scores, key=lambda x: x[2], reverse=True)
        anomalies = [s for s in sorted_scores if round(s[2]/1000000) > threshold]

        if anomalies:
            log.info(f"{len(anomalies)} anomalies was detected on sli-rtt model.")

        return anomalies[:3], sorted_scores[:1]

    def detect_tps(self, utc_now: datetime, machine_id: str):
        """Detects tps by rule based model"""
        tim_start = utc_now - timedelta(minutes=10)
        tim_end = utc_now

        metric_name = self.tps.name
        threshold = self.tps.threshold
        metric_loader = MetricLoader(tim_start, tim_end, self.config)
        labels, values = metric_loader.get_metric(metric_name, label_name="machine_id", label_value=machine_id)

        point_count = metric_loader.expected_point_length()

        scores = []
        for lbl, val in zip(labels, values):
            if lbl.get("datname", "") != "postgres":
                continue

            if len(val) < point_count * 0.9 or len(val) > point_count * 1.5:
                continue

            tps_val = np.array([np.float64(v[1]) for v in val])

            tps_val = smooth(tps_val, box_pts=13)
            score = tps_val[: -13] / tps_val[13:]
            score = max(score[-12:])

            if math.isnan(score) or math.isinf(score):
                continue

            scores.append((lbl["__name__"], lbl, score))

        sorted_scores = sorted(scores, key=lambda x: x[2], reverse=True)
        anomalies = [s for s in sorted_scores if s[2] >= threshold]

        if anomalies:
            log.info(f"{len(anomalies)} anomalies was detected on sli-tps model.")

        return anomalies
