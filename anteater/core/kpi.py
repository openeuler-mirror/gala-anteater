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

from dataclasses import dataclass, field
from typing import List, Optional

from anteater.core.anomaly import AnomalyTrend


@dataclass
class KPI:
    """The kpi for an abstract target of AD job"""
    metric: str
    entity_name: str
    description: str = ""
    enable: bool = True
    params: dict = field(default=dict)
    atrend: AnomalyTrend = AnomalyTrend.DEFAULT

    @classmethod
    def from_dict(cls, **data):
        if 'atrend' in data:
            data['atrend'] = AnomalyTrend.from_str(data.get('atrend'))

        return cls(**data)


@dataclass
class Feature:
    """The feature with it's detailed infos"""
    metric: str
    description: str = ""
    priority: int = 0
    atrend: AnomalyTrend = AnomalyTrend.DEFAULT

    @classmethod
    def from_dict(cls, **data):
        """Converts data to Feature object"""
        if 'atrend' in data:
            data['atrend'] = AnomalyTrend.from_str(data.get('atrend'))

        return cls(**data)


@dataclass
class ModelConfig:
    """The model config for a specific ML model"""
    name: str
    model_path: str
    params: dict = field(default=dict)


@dataclass
class JobConfig:
    """The Job Config would be passed to an AD job"""
    name: str
    enable: bool
    job_type: str
    detector: str
    template: str
    keywords: List[str]
    root_cause_num: int
    kpis: List[KPI]
    features: List[Feature]
    model_config: Optional[ModelConfig]
