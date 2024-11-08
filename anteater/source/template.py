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

from typing import List, Dict
from anteater.core.anomaly import Anomaly
from anteater.utils.datetime import DateTimeManager as dt
import time

class Template:
    """The anomaly events template"""
    def __init__(self, **kwargs):
        self._machine_id = ''
        self._metric = ''
        self._entity_name = ''
        self._score = -1
        self._root_causes = []

        self._labels = {}
        self._entity_id = ""
        self._keywords = []

        self._description = ""
        self._details = {}

        self.header = None
        self.event_type = None
        self._timestamp = None

        # 新增异常字段, 默认调用该模板上传kafka为异常事件
        self.is_anomaly = True
        self._cause_metrics_list = []

        self.anomaly_cluster_info = {}

    def add_anomaly_cluster_info(self, ab_info: str):
        self.anomaly_cluster_info = ab_info

    def get_template(self):
        """Gets the template for app level anomaly events"""
        self._timestamp = dt.utc_now()
        round_timestamp = round(self._timestamp.timestamp() * 1000)

        #
        result = {
            'Timestamp': round_timestamp,
            'Attributes': {
                'entity_id': self._entity_id,
                'event_id': f'{round_timestamp}_{self._entity_id}',
                'event_type': self.event_type,
                'event_source': 'gala-anteater',
                'keywords': self._keywords,
                'cause_metric': self._cause_metrics_list[0] if self._cause_metrics_list else {
                    'description': self._description}
            },
            'Resource': {
                'metric': self._metric,
                'labels': self._labels,
                'score': f'{self._score:.3f}',
                'root_causes': self._cause_metrics_list,
                'description': self._description
            },
            'SeverityText': 'WARN',
            'SeverityNumber': 13,
            'Body': self.get_body(),
            "is_anomaly": self.is_anomaly
        }

        return result

    def parse_anomaly(self, anomaly: Anomaly):
        """Parses the anomaly object properties"""
        self._machine_id = anomaly.machine_id
        self._metric = anomaly.metric
        self._entity_name = anomaly.metric
        self._score = anomaly.score
        self._root_causes = anomaly.root_causes
        self._cause_metrics_list = [
            {
                'metric': cause.metric,
                'labels': cause.labels,
                'score': f'{cause.score:.3f}',
            }
            for cause in anomaly.root_causes]

    def add_labels(self, labels):
        """Adds labels property"""
        self._labels = labels

    def add_entity_id(self, entity_id):
        """Adds entity id property"""
        self._entity_id = entity_id

    def add_anomaly_status(self, is_anomaly):
        """Adds entity id property"""
        self.is_anomaly = is_anomaly

    def add_keywords(self, keywords: List[str]):
        """Adds keywords property"""
        self._keywords = keywords

    def add_description(self, description: str):
        """Add description property"""
        self._description = description

    def add_details(self, details: dict):
        """Adds details property in the body"""
        self._details = details



    def get_body(self) -> str:
        """Gets the template body property"""
        if not self.header:
            raise ValueError('Non-initialized property \'header\' on '
                             f'{self.__class__.__name__}.')
        body = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._timestamp.timestamp())),
                self.header, self._description, self._details]

        return ' - '.join([str(x) for x in body if x])

    def get_root_causes(self) -> List:
        """Gets root causes"""
        if not self._root_causes:
            return []

        return [str(x) for x in self._root_causes]


class AppAnomalyTemplate(Template):
    """The app anomaly template"""

    def __init__(self, **kwargs):
        """The app anomaly template initializer"""
        super().__init__(**kwargs)
        self.header = "Application Failure"
        self.event_type = "app"


class NetAnomalyTemplate(Template):
    """The app anomaly template"""

    def __init__(self, **kwargs):
        """The app anomaly template initializer"""
        super().__init__(**kwargs)
        self.header = "Neural network Detection"
        self.event_type = "net"

    def parse_anomaly(self, anomaly: Anomaly):
        """Parses the anomaly object properties"""
        self._machine_id = anomaly.machine_id
        self._metric = anomaly.metric
        self._entity_name = anomaly.metric
        self._score = anomaly.score
        self._root_causes = anomaly.root_causes
        self._cause_metrics_list = [
            {
                'metric': cause.ts.metric,
                'labels': cause.ts.labels,
                'score': f'{cause.score:.3f}',
            }
            for cause in anomaly.root_causes]


class SysAnomalyTemplate(Template):
    """The sys anomaly template"""

    def __init__(self, **kwargs):
        """The sys anomaly template initializer"""
        super().__init__(**kwargs)
        self.header = 'System Failure'
        self.event_type = 'sys'


class JVMAnomalyTemplate(Template):
    """The jvm anomaly template"""

    def __init__(self, **kwargs):
        """The jvm anomaly template initializer"""
        super().__init__(**kwargs)
        self.header = "JVM OutOfMemory"
        self.event_type = 'jvm'


class SimpleAnomalyTemplate(Template):
    """The jvm anomaly template"""

    def __init__(self, **kwargs):
        """The simple anomaly template initializer"""
        super().__init__(**kwargs)
        self.header = "app metric anomaly"
        self.event_type = "simple"

    def get_template(self):
        """Gets the template for app level anomaly events"""
        self._timestamp = dt.utc_now()
        round_timestamp = round(self._timestamp.timestamp() * 1000)

        event_source = self._details.get('event_source', 'gala-anteater')

        #
        result = {
            'Timestamp': round_timestamp,
            'Attributes': {
                'entity_id': self._entity_id,
                'event_id': f'{round_timestamp}_{self._entity_id}',
                'event_type': self.event_type,
                'event_source': event_source,
                'keywords': self._keywords,
            },
            'Resource': {
                'metric': self._metric,
                'labels': self._labels,
                'score': f'{self._score:.3f}',
                'root_causes': self._cause_metrics_list,
                'description': self._description
            },
            'SeverityText': 'WARN',
            'SeverityNumber': 13,
            'Body': self.get_body(),
            "is_anomaly": self.is_anomaly
        }

        return result


class SlowNodeTemplate(Template):
    """The jvm anomaly template"""

    def __init__(self, **kwargs):
        """The simple anomaly template initializer"""
        super().__init__(**kwargs)
        self.header = "slow node"
        self.event_type = "slow_node"


    def parse_anomaly(self, anomaly: Anomaly):
        """Parses the anomaly object properties"""
        self._machine_id = anomaly.machine_id
        self._metric = anomaly.metric
        self._entity_name = anomaly.metric
        self._score = anomaly.score
        self._root_causes = anomaly.root_causes


    def get_template(self):
        """Gets the template for app level anomaly events"""
        self._timestamp = dt.utc_now()
        round_timestamp = round(self._timestamp.timestamp() * 1000)

        result = {
            'Timestamp': round_timestamp,
            'Attributes': self.anomaly_cluster_info,
            'SeverityText': 'WARN',
            'SeverityNumber': 13,
            "is_anomaly": self.is_anomaly
        }

        return result