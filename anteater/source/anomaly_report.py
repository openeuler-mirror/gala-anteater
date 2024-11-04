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

import re
from typing import Dict, List

from anteater.core.anomaly import Anomaly
from anteater.core.info import MetricInfo
from anteater.provider.kafka import KafkaProvider
from anteater.source.suppress import AnomalySuppression
from anteater.source.template import Template
from anteater.utils.constants import COMM, CONTAINER_ID,\
    DEV_NAME, DEVICE, DISK_NAME, FSNAME, IP, MACHINE_ID,\
    PID, POD_NAME, SERVER_IP, TGID
from anteater.utils.log import logger

PUNCTUATION_PATTERN = re.compile(r"[^\w_\-:.@()+,=;$!*'%]")


class AnomalyReport:
    """The anomaly events report class

    Which will send anomaly events to the specified provider,
    currently, we send the anomaly events to the Kafka.
    """
    def __init__(
            self,
            provider: KafkaProvider,
            suppressor: AnomalySuppression,
            metricinfo: MetricInfo):
        """The Anomaly Report class initializer"""
        self.provider = provider
        self.suppressor = suppressor
        self.metricinfo = metricinfo

    @staticmethod
    def extract_entity_id(machine_id, entity_name, labels, keys):
        """Extracts entity id on the machine id, entity name, label and keys"""
        label_keys = [labels.get(k, '0') for k in keys if k != "machine_id"]
        entity_id = f'{machine_id}_{entity_name}_{"_".join(label_keys)}'
        entity_id = PUNCTUATION_PATTERN.sub(':', entity_id)

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

        keys = ['Host', 'PID', 'COMM', 'IP', 'ContainerID', 'POD', 'Device', 'ContainerName', 'ContainerHost']
        values = [
            f'{machine_id}-{ip}' if machine_id and ip else '',
            labels.get(TGID, '') or labels.get(PID, ''),
            labels.get(COMM, ''),
            ip,
            labels.get(CONTAINER_ID, ''),
            labels.get(POD_NAME, ''),
            (labels.get(DEVICE, '') or labels.get(DEV_NAME, '') or
             labels.get(DISK_NAME, '') or labels.get(FSNAME, '')),
            labels.get('container_image', ''),
            labels.get('container_hostname', '')
        ]

        return dict([(k, v) for k, v in zip(keys, values) if v])

    def extract_keys(self, entity_name):
        """Extracts keys from the specific metadata"""
        keys = self.provider.get_metadata(entity_name)

        if not keys:
            logger.warning('Empty metadata for entity name %s!', entity_name)

        return keys

    def get_description(self, metric) -> str:
        """Gets the description based on the metric name"""
        des = self.metricinfo.get_zh(metric)
        if not des:
            des = self.metricinfo.get_en(metric)

        return des

    def sent_anomaly(self, anomaly: Anomaly,
                     keywords: List[str], template: Template) -> None:
        """Sends the anomaly events to the provider"""
        if self.suppressor.suppress(anomaly):
            return

        keys = self.extract_keys(anomaly.entity_name)
        labels = self.purify_labels(anomaly.labels)
        entity_id = self.extract_entity_id(anomaly.machine_id,
                                           anomaly.entity_name,
                                           anomaly.labels, keys)
        description = self.get_description(anomaly.metric)

        template.parse_anomaly(anomaly)
        template.add_labels(labels)
        template.add_entity_id(entity_id)
        template.add_keywords(keywords)
        template.add_details(details=anomaly.details)
        template.add_description(description=description)
        template.add_anomaly_status(anomaly.is_anomaly)
        template.add_anomaly_cluster_info(anomaly.description)

        msg = template.get_template()
        self.provider.send_message(msg)
