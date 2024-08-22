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
from typing import Any, Dict


from anteater.utils.log import logger

from kafka import KafkaConsumer, KafkaProducer
from kafka import TopicPartition as tp


class KafkaProvider:
    """The Kafka provider provides consuming and producing
    messages from Kafka service
    """

    def __init__(self, conf) -> None:
        producer_configs = {
            "bootstrap_servers": f"{conf.server}:{conf.port}",
            "api_version": (0, 10, 2)
        }
        consumer_configs = {
            "bootstrap_servers": f"{conf.server}:{conf.port}",
            "auto_offset_reset": "earliest",
            "enable_auto_commit": False,
            "consumer_timeout_ms": 1000,
            "group_id": conf.group_id,
            "api_version": (0, 10, 2)
        }
        self.topic = conf.model_topic
        self.rca_topic = conf.rca_topic
        self.producer = KafkaProducer(**producer_configs)
        self.consumer = KafkaConsumer(self.topic, **consumer_configs)
        self.metadata = collections.deque(maxlen=200)
        self.updating()

    def updating(self):
        self.fetch_metadata()

    def fetch_metadata(self):
        index = 1
        for msg in self.consumer:
            index += 1
            data = json.loads(msg.value)
            metadata = {}
            metadata.update(data)
            self.metadata.append(metadata)

    def get_metadata(self, entity_name):
        for item in self.metadata:
            if item.get("entity_name", "") == entity_name:
                return item.get("keys", {})
        logger.error(f"Unknown entity_name {entity_name} in metadata")
        return {}

    def send_message(self, message: Dict[str, Any]):
        """Sent the message to Kafka"""
        # logger.logger.info(f"Sending the inference message to Kafka: {str(message)}")
        self.producer.send(self.rca_topic, json.dumps(message).encode('utf-8'))
        self.producer.flush()

    def range_query(self, start_time, end_time):
        start_ms = round(start_time.timestamp() * 1000)
        end_ms = round(end_time.timestamp() * 1000)

        result = []
        for p in self.consumer.partitions_for_topic(self.topic):
            start_offset = self.consumer.offsets_for_times({tp(self.topic, p): start_ms})

            if not start_offset or not start_offset[tp(self.topic, p)]:
                continue

            start_offset = start_offset[tp(self.topic, p)].offset
            end_offset = self.consumer.end_offsets([tp(self.topic, p)])[tp(self.topic, p)]

            self.consumer.seek(tp(self.topic, p), start_offset)

            for record in self.consumer:
                if record.offset >= end_offset or record.timestamp > end_ms:
                    break
                data = json.loads(record.value)
                result.append(data)
        return result
