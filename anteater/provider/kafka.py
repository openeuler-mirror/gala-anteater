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
Description: The implementation of Kafka Consumer and Producer.
"""

import collections
import json
import threading
import time
from typing import Any, Dict

from kafka import KafkaConsumer, KafkaProducer

from anteater.config import KafkaConf
from anteater.utils.log import logger


class EntityVariable:
    """
    The global variables which will be used to update
    some key settings through multiprocessors
    """
    variable = None


class KafkaProvider:
    """The Kafka provider provides consuming and producing
    messages from Kafka service
    """

    def __init__(self, conf: KafkaConf) -> None:
        self.conf = conf
        producer_configs = {
            "bootstrap_servers": f"{conf.server}:{conf.port}"
        }
        consumer_configs = {
            "bootstrap_servers": f"{conf.server}:{conf.port}",
            "auto_offset_reset": "earliest",
            "enable_auto_commit": False,
            "consumer_timeout_ms": 1000,
            "group_id": conf.group_id,
        }

        if conf.auth_type == 'sasl_plaintext':
            self.config_kafka_sasl(producer_configs)
            self.config_kafka_sasl(consumer_configs)

        self.model_topic = conf.model_topic
        self.meta_topic = conf.meta_topic

        self.producer = KafkaProducer(**producer_configs)
        self.consumer = KafkaConsumer(self.meta_topic, **consumer_configs)

        self.metadata = collections.deque(maxlen=200)
        self.updating()

    def config_kafka_sasl(self, kafka_conf):
        """Config kafka sasl plaintext"""
        kafka_conf['security_protocol'] = "SASL_PLAINTEXT"
        kafka_conf['sasl_mechanism'] = "PLAIN"
        kafka_conf['sasl_plain_username'] = self.conf.username
        kafka_conf['sasl_plain_password'] = self.conf.password

    def updating(self):
        t = threading.Thread(target=self.fetch_metadata, args=())
        t.start()

    def fetch_metadata(self):
        while True:
            for msg in self.consumer:
                data = json.loads(msg.value)
                metadata = {}
                metadata.update(data)
                self.metadata.append(metadata)
            time.sleep(5)

    def get_metadata(self, entity_name):
        for item in self.metadata:
            if item.get("entity_name", "") == entity_name:
                return item.get("keys", {})
        logger.error(f"Unknown entity_name {entity_name} in metadata")
        return {}

    def send_message(self, message: Dict[str, Any]):
        """Sent the message to Kafka"""
        logger.info(f"Sending the abnormal message to Kafka: {str(message)}")
        self.producer.send(self.model_topic, json.dumps(message).encode('utf-8'))
        self.producer.flush()
