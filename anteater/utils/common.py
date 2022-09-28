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
Description: Some common functions are able to use in this project.
"""

import os
import re
import stat
from datetime import datetime
from typing import Dict, Any, List, Tuple

import yaml

from anteater.config import AnteaterConf
from anteater.provider.kafka import KafkaConsumer, EntityVariable
from anteater.utils.data_load import load_metric_description
from anteater.utils.log import logger


PUNCTUATION_PATTERN = re.compile(r"[^\w_\-:.@()+,=;$!*'%]")


def load_model_conf(filename):
    """Loads config and build objects"""
    root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    file_path = os.path.join(root_path, "config" + os.sep + filename)

    if not os.path.isfile(file_path):
        logger.warning(f"Anteater config file was not found in the folder: {root_path}!")
        return {}

    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(file_path, os.O_RDONLY, modes), "r") as f_out:
        parameters = yaml.safe_load(f_out)

    return parameters


def get_file_path(file_name):
    """Gets root path of anteater"""
    root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    file_path = os.path.join(root_path, "file" + os.sep + file_name)

    return file_path


def update_metadata(config: AnteaterConf) -> KafkaConsumer:
    """Updates entity variables by querying data from Kafka under sub thread"""
    logger.info("Start to try updating global configurations by querying data from Kafka!")

    server = config.kafka.server
    port = config.kafka.port
    topic = config.kafka.meta_topic

    entity_name = config.kafka.meta_entity_name

    consumer = KafkaConsumer(server, port, topic, entity_name)
    consumer.start()

    return consumer


def get_kafka_message(utc_now: datetime, y_pred: List, machine_id: str, key_anomalies: Tuple[str, Dict, float],
                      rec_anomalies: List[Tuple[str, Dict, float]]) -> Dict[str, Any]:
    """Generates the Kafka message based the parameters"""
    variable = EntityVariable.variable.copy()

    entity_name = variable["entity_name"]
    filtered_metric_label = {}
    keys = []

    metric_label = key_anomalies[1]
    metric_id = key_anomalies[0]

    for key in variable["keys"]:
        filtered_metric_label[key] = metric_label.get(key, "0")
        if key != "machine_id":
            keys.append(metric_label.get(key, "0"))

    entity_id = f"{machine_id}_{entity_name}_{'_'.join(keys)}"
    entity_id = PUNCTUATION_PATTERN.sub(":", entity_id)

    sample_count = len(y_pred)
    if sample_count != 0:
        anomaly_score = sum(y_pred) / sample_count
    else:
        anomaly_score = 0

    recommend_metrics = list()
    descriptions = load_metric_description()
    for name, label, score in rec_anomalies:
        recommend_metrics.append({
            "metric": name,
            "label": label,
            "score": score,
            "description": descriptions.get(name, "")
        })

    timestamp = round(utc_now.timestamp() * 1000)

    message = {
        "Timestamp": timestamp,
        "Attributes": {
            "entity_id": entity_id,
            "event_id": f"{timestamp}_{entity_id}",
            "event_type": "app"
        },
        "Resource": {
            "anomaly_score": anomaly_score,
            "anomaly_count": sum(y_pred),
            "total_count": len(y_pred),
            "duration": 60,
            "anomaly_ratio": anomaly_score,
            "metric_label": filtered_metric_label,
            "recommend_metrics": recommend_metrics,
            "metrics": metric_id,
            "description": descriptions.get(metric_id, "")
        },
        "SeverityText": "WARN",
        "SeverityNumber": 13,
        "Body": f"{utc_now.strftime('%c')}, WARN, APP may be impacting sli performance issues.",
        "event_id": f"{timestamp}_{entity_id}"
    }

    return message


def get_sys_message(utc_now: datetime, metric_id: str, machine_id: str) -> Dict[str, Any]:
    """Generates the Kafka sys message based the parameters"""
    entity_id = machine_id
    entity_id = PUNCTUATION_PATTERN.sub(":", entity_id)

    timestamp = round(utc_now.timestamp() * 1000)

    message = {
        "Timestamp": timestamp,
        "Attributes": {
            "entity_id": entity_id,
            "event_id": f"{timestamp}_{entity_id}",
            "event_type": "sys"
        },
        "Resource": {
            "metrics": metric_id,
        },
        "SeverityText": "WARN",
        "SeverityNumber": 13,
        "Body": f"{utc_now.strftime('%c')}, WARN, Sys metric {entity_id} may be impacting some issues.",
        "event_id": f"{timestamp}_{entity_id}"
    }

    return message
