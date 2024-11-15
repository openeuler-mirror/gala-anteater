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

from dataclasses import dataclass
from enum import Enum
from typing import List


@dataclass
class RootCause:
    """The root cause of an anomaly event"""
    metric: str
    labels: dict
    score: float

    def __str__(self) -> str:
        dic = {
            "metric": self.metric,
            "labels": self.labels,
            "score": f'{self.score:.3f}'
        }
        return str(dic)


@dataclass
class Anomaly:
    """The anomaly events dataclass"""
    machine_id: str
    metric: str
    labels: dict
    score: float
    entity_name: str
    description: str = ""
    details: dict = None
    is_anomaly: bool = True
    root_causes: List[RootCause] = None

    def __eq__(self, other):
        if not other or not isinstance(other, Anomaly):
            return False

        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def id(self):
        """The unique id for the anomaly"""
        return self.metric + str(tuple(sorted(self.labels.items())))


class AnomalyTrend(Enum):
    """The anomaly trend of the kpi
    - such as: a larger TCP link srtt probably would be thought
              as an anomaly event, usually, a smaller one should be
              thought as a normal event.
    """
    DEFAULT = 0
    RISE = 1
    FALL = 2

    @staticmethod
    def from_str(label: str):
        """Trans str to Enum type"""
        if label.upper() == 'RISE':
            return AnomalyTrend.RISE
        elif label.upper() == 'FALL':
            return AnomalyTrend.FALL
        elif label.upper() == "DEFAULT":
            return AnomalyTrend.DEFAULT
        else:
            raise ValueError(f'Unknown anomaly trend type: {label}')
