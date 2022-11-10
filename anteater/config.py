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
"""
Time:
Author:
Description: The gala-anteater global configurations.
"""

import logging
import os.path
from dataclasses import dataclass, field

import yaml


@dataclass
class GlobalConf:
    """The global config"""
    data_source: str = None


@dataclass
class ServiceConf:
    """The provider config"""
    server: str = None
    port: str = None
    steps: int = None


@dataclass
class KafkaConf(ServiceConf):
    """The kafka config"""
    model_topic: str = None
    model_group_id: str = None
    meta_topic: str = None
    meta_group_id: str = None
    meta_entity_name: str = None


@dataclass
class PrometheusConf(ServiceConf):
    """The prometheus config"""
    step: int = None


@dataclass
class AomConfig:
    base_url: str = None
    project_id: str = None
    auth_type: str = None
    auth_info: dict = field(default_factory=dict)


@dataclass
class HybridModelConfig:
    """The hybrid model config"""
    name: str = None
    model_folder: str = None
    latest_model_folder: str = None
    look_back: int = None
    threshold: float = None
    retrain: bool = False
    keep_model: bool = True
    use_latest_model: bool = True


@dataclass
class ScheduleConf:
    """The scheduling method config"""
    duration: int = None


class AnteaterConf:
    """The gala-anteater globally configurations"""

    filename = "gala-anteater.yaml"

    def __init__(self):
        """The gala-anteater config initializer"""
        self.global_conf: GlobalConf = None
        self.kafka: KafkaConf = None
        self.prometheus: PrometheusConf = None
        self.aom: AomConfig = None
        self.hybrid_model: HybridModelConfig = None
        self.schedule: ScheduleConf = None

    def load_from_yaml(self, data_path: str):
        """Loads config from yaml file"""
        data_path = os.path.realpath(data_path)

        if not os.path.exists(data_path):
            os.makedirs(data_path)

        try:
            with open(os.path.join(data_path, "config", self.filename), "rb") as f:
                result = yaml.safe_load(f)
        except IOError as e:
            logging.error(f"Load gala-anteater config file failed {e}")
            raise e

        global_conf = result.get("Global")
        kafka_conf = result.get("Kafka")
        prometheus_conf = result.get("Prometheus")
        aom_config = result.get("Aom")
        hybrid_model_conf = result.get("HybridModel")
        schedule_conf = result.get("Schedule")

        data_source = global_conf.get("data_source")

        self.global_conf = GlobalConf(data_source=data_source)
        self.kafka = KafkaConf(**kafka_conf)
        self.prometheus = PrometheusConf(**prometheus_conf)
        self.aom = AomConfig(**aom_config)
        self.hybrid_model = HybridModelConfig(**hybrid_model_conf)
        self.hybrid_model.model_folder = os.path.join(data_path, self.hybrid_model.model_folder)
        self.hybrid_model.latest_model_folder = os.path.join(data_path, self.hybrid_model.latest_model_folder)
        self.schedule = ScheduleConf(**schedule_conf)
