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
from typing import List

from anteater.core.anomaly import Anomaly
from anteater.provider.kafka import KafkaProvider
from anteater.template.template import Template
from anteater.utils.log import logger

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

    def get_keys(self, entity_name):
        keys = self.provider.get_metadata(entity_name)

        if not keys:
            logger.warning(f"Empty metadata for entity name {entity_name}!")

        return keys

    def sent_anomaly(self, anomaly: Anomaly, cause_metrics: List, keywords: List[str], template: Template):
        keys = self.get_keys(template.entity_name)
        machine_id = template.machine_id
        entity_name = template.entity_name
        labels = anomaly.labels

        template.score = anomaly.score
        template.labels = labels
        template.entity_id = self.get_entity_id(machine_id, entity_name, labels, keys)
        template.keys = keys
        template.description = anomaly.description
        template.cause_metrics = cause_metrics
        template.keywords = keywords

        msg = template.get_template()
        self.provider.send_message(msg)
