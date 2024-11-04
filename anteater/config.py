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
"""
Time:
Author:
Description: The gala-anteater global configurations.
"""

import os.path
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class GlobalConf:
    """The global config"""
    data_source: str


@dataclass
class ServiceConf:
    """The provider config"""
    server: str
    port: str

@dataclass
class ArangodbConf:
    """The Arangodb config"""
    url: str
    db_name: str

@dataclass
class KafkaConf(ServiceConf):
    """The kafka config"""
    model_topic: str
    rca_topic: str
    meta_topic: str
    group_id: str
    auth_type: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None


@dataclass
class PrometheusConf(ServiceConf):
    """The prometheus client config"""
    steps: int


@dataclass
class AomConfig:
    """The aom client config"""
    base_url: str
    project_id: str
    auth_type: str
    auth_info: dict = field(default_factory=dict)


@dataclass
class ScheduleConf:
    """The scheduling method config"""
    duration: int


class AnteaterConf:
    """The gala-anteater globally configurations"""

    filename = 'gala-anteater.yaml'

    def __init__(self):
        """The gala-anteater config initializer"""
        self.global_conf: GlobalConf = None
        self.kafka: KafkaConf = None
        self.prometheus: PrometheusConf = None
        self.aom: AomConfig = None
        self.schedule: ScheduleConf = None
        self.arangodb: ArangodbConf = None
        self.suppression_time: int = 5

    def load_from_yaml(self, data_path: str):
        """Loads config from yaml file"""
        data_path = os.path.realpath(data_path)

        try:
            with open(os.path.join(data_path, self.filename), 'rb') as f:
                result = yaml.safe_load(f)
        except IOError as e:
            raise ValueError('Load gala-anteater config file failed') from e

        global_conf = result.get('Global')
        kafka_conf = result.get('Kafka')
        prometheus_conf = result.get('Prometheus')
        aom_config = result.get('Aom')
        schedule_conf = result.get('Schedule')
        arangodb_conf = result.get('Arangodb')
        suppression = result.get('Suppression', {})

        self.global_conf = GlobalConf(**global_conf)
        self.kafka = KafkaConf(**kafka_conf)
        self.prometheus = PrometheusConf(**prometheus_conf)
        self.aom = AomConfig(**aom_config)
        self.schedule = ScheduleConf(**schedule_conf)
        self.arangodb = ArangodbConf(**arangodb_conf)
        self.suppression_time = suppression.get("interval", 5)