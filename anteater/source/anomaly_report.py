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

import logging
import re
from typing import Dict, List

from anteater.core.anomaly import Anomaly
from anteater.provider.kafka import KafkaProvider
from anteater.template.template import Template
from anteater.utils.constants import COMM, CONTAINER_ID,\
    DEV_NAME, DEVICE, DISK_NAME, FSNAME, IP, MACHINE_ID,\
    PID, POD_NAME, SERVER_IP, TGID

PUNCTUATION_PATTERN = re.compile(r"[^\w_\-:.@()+,=;$!*'%]")


class AnomalyReport:
    def __init__(self, provider: KafkaProvider):
        self.provider = provider

    @staticmethod
    def get_entity_id(machine_id, entity_name, labels, keys):
        label_keys = [labels.get(key, '0') for key in keys if key != "machine_id"]
        entity_id = f"{machine_id}_{entity_name}_{'_'.join(label_keys)}"
        entity_id = PUNCTUATION_PATTERN.sub(":", entity_id)

        return entity_id

    @staticmethod
    def purify_labels(labels: Dict) -> Dict:
        """Purifying raw labels to keep specific keys, includes:
        'Host', 'PID', 'COMM', 'IP', 'ContainerID', 'Device', etc.
        """
        if not labels:
            return {}

        if not isinstance(labels, dict):
            raise TypeError('The type of labels is not a dict.')

        machine_id = labels.get(MACHINE_ID, '')
        ip = labels.get(IP, '') or labels.get(SERVER_IP, '')

        keys = ['Host', 'PID', 'COMM', 'IP', 'ContainerID', 'POD', 'Device']
        values = [
            f'{machine_id}-{ip}' if machine_id and ip else '',
            labels.get(TGID, '') or labels.get(PID, ''),
            labels.get(COMM, ''),
            ip,
            labels.get(CONTAINER_ID, ''),
            labels.get(POD_NAME, ''),
            (labels.get(DEVICE, '') or labels.get(DEV_NAME, '') or
             labels.get(DISK_NAME, '') or labels.get(FSNAME, ''))
        ]

        return dict([(k, v) for k, v in zip(keys, values) if v])

    def get_keys(self, entity_name):
        keys = self.provider.get_metadata(entity_name)

        if not keys:
            logging.warning(f"Empty metadata for entity name {entity_name}!")

        return keys

    def sent_anomaly(self, anomaly: Anomaly, cause_metrics: List, keywords: List[str], template: Template):
        keys = self.get_keys(template.entity_name)
        machine_id = template.machine_id
        entity_name = template.entity_name
        labels = anomaly.labels

        template.score = anomaly.score
        template.labels = self.purify_labels(labels)
        template.entity_id = self.get_entity_id(machine_id, entity_name, labels, keys)
        template.description = anomaly.description
        template.cause_metrics = cause_metrics
        template.keywords = keywords

        msg = template.get_template()
        self.provider.send_message(msg)
