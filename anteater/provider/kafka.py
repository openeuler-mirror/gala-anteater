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
from datetime import datetime

from kafka import KafkaConsumer, KafkaProducer, TopicPartition

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
            "bootstrap_servers": f"{conf.server}:{conf.port}",
            "api_version": (0, 10, 2)
        }
        consumer_configs = {
            "bootstrap_servers": f"{conf.server}:{conf.port}",
            "auto_offset_reset": "earliest",
            "enable_auto_commit": False,
            "consumer_timeout_ms": 1000,
            "api_version": (0, 10, 2)
        }

        if conf.auth_type == 'sasl_plaintext':
            self.config_kafka_sasl(producer_configs)
            self.config_kafka_sasl(consumer_configs)

        self.model_topic = conf.model_topic
        self.rca_topic = conf.rca_topic
        self.meta_topic = conf.meta_topic

        self.producer = KafkaProducer(**producer_configs)
        self.consumer_meta = KafkaConsumer(self.meta_topic, **consumer_configs)
        self.consumer_model = KafkaConsumer(self.model_topic, **consumer_configs)

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
            for msg in self.consumer_meta:
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
        self.producer.send(self.model_topic, json.dumps(message).encode('utf-8'))
        logger.info(f"send anomaly message to kafka.")
        self.producer.flush()

    def send_rca_message(self, message: Dict[str, Any]):
        """Sent the message to Kafka"""
        # logger.info(f"Sending the abnormal message to Kafka: {str(message)}")
        self.producer.send(self.rca_topic, json.dumps(message).encode('utf-8'))
        self.producer.flush()

    def range_query(self, start_time: datetime, end_time: datetime) -> list:
        start_ms = round(start_time.timestamp() * 1000)
        end_ms = round(end_time.timestamp() * 1000)

        result = []
        for p in self.consumer_model.partitions_for_topic(self.model_topic):
            start_offset = self.consumer_model.offsets_for_times({TopicPartition(self.model_topic, p): start_ms})

            if not start_offset or not start_offset[TopicPartition(self.model_topic, p)]:
                continue

            start_offset = start_offset[TopicPartition(self.model_topic, p)].offset
            end_offset = self.consumer_model. \
                end_offsets([TopicPartition(self.model_topic, p)])[TopicPartition(self.model_topic, p)]

            self.consumer_model.seek(TopicPartition(self.model_topic, p), start_offset)

            for record in self.consumer_model:
                if record.offset >= end_offset or record.timestamp > end_ms:
                    break
                data = json.loads(record.value)
                result.append(data)
        return result
