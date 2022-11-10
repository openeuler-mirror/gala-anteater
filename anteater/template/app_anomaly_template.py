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

from anteater.template.template import Template


class AppAnomalyTemplate(Template):
    """The app anomaly template"""
    def __init__(self, timestamp, machine_id, metric_id, entity_name):
        super().__init__(timestamp, machine_id, metric_id, entity_name)

    def get_template(self):
        timestamp = round(self.timestamp.timestamp() * 1000)

        result = {
            "Timestamp": timestamp,
            "Attributes": {
                "entity_id": self.entity_id,
                "event_id": f"{timestamp}_{self.entity_id}",
                "event_type": "app"
            },
            "Resource": {
                "metrics": self.metric_id,
                "metric_label": self.labels,
                "recommend_metrics": self.cause_metrics,
                "description": self.description
            },
            "SeverityText": "WARN",
            "SeverityNumber": 13,
            "Body": f"{self.timestamp.strftime('%c')} WARN, APP may be impacting sli performance issues.",
            "event_id": f"{timestamp}_{self.entity_id}",
        }

        return result
