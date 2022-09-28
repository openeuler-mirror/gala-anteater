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
Description: The gala-anteater global configurations.
"""

import os.path
from dataclasses import dataclass

import yaml

from anteater.utils.log import Log

log = Log().get_logger()


@dataclass
class Service:
    """The service config"""
    server: str = None
    port: str = None
    steps: int = None


@dataclass
class Kafka(Service):
    """The kafka config"""
    model_topic: str = None
    model_group_id: str = None
    meta_topic: str = None
    meta_group_id: str = None
    meta_entity_name: str = None


@dataclass
class Prometheus(Service):
    """The prometheus config"""
    step: int = None


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
class RTT:
    """The rtt config in sli-model"""
    name: str = None
    threshold: float = None


@dataclass
class TPS:
    """The tps config in sli-model"""
    name: str = None
    threshold: float = None


@dataclass
class SLIModel:
    """The sli model config"""
    rtt: RTT = None
    tps: TPS = None


@dataclass
class ScheduleConfig:
    """The scheduling method config"""
    duration: int = None


class AnteaterConfig:
    """The gala-anteater globally configurations"""

    filename = "gala-anteater.yaml"

    def __init__(
        self,
        kafka: Kafka,
        prometheus: Prometheus,
        hybrid_model: HybridModelConfig,
        sli_model: SLIModel,
        schedule: ScheduleConfig
    ):
        """The gala-anteater config initializer"""
        self.kafka: Kafka = kafka
        self.prometheus: Prometheus = prometheus
        self.hybrid_model = hybrid_model
        self.sli_model = sli_model
        self.schedule = schedule

    @classmethod
    def load_from_yaml(cls, data_path: str):
        """Loads config from yaml file"""
        data_path = os.path.realpath(data_path)

        if not os.path.exists(data_path):
            os.makedirs(data_path)

        try:
            with open(os.path.join(data_path, "config", cls.filename), "rb") as f:
                result = yaml.safe_load(f)
        except IOError as e:
            log.error(f"Load gala-anteater config file failed {e}")
            raise e

        kafka_conf = result.get("Kafka")
        prometheus_conf = result.get("Prometheus")
        hybrid_model_conf = result.get("HybridModel")
        sli_model_conf = result.get("SLIModel")
        schedule_conf = result.get("SCHEDULE")

        rtt = sli_model_conf.get("RTT")
        tps = sli_model_conf.get("TPS")

        kafka = Kafka(**kafka_conf)
        prometheus = Prometheus(**prometheus_conf)
        hybrid_model = HybridModelConfig(**hybrid_model_conf)
        sli_model = SLIModel(rtt=RTT(**rtt), tps=TPS(**tps))
        schedule = ScheduleConfig(**schedule_conf)

        hybrid_model.model_folder = os.path.join(data_path, hybrid_model.model_folder)
        hybrid_model.latest_model_folder = os.path.join(data_path, hybrid_model.latest_model_folder)

        return cls(kafka, prometheus, hybrid_model, sli_model, schedule)
