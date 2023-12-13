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

class Entity:
    """The entity base class"""
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


class AppEntity(Entity):
    """The app entity"""

    def __init__(self, **kwargs):
        """The app entity initializer"""
        super().__init__(**kwargs)
        self.header = "Application Failure"
        self.event_type = "app"


class ServiceEntity(Entity):
    """The service entity"""

    def __init__(self, **kwargs):
        """The service entity initializer"""
        super().__init__(**kwargs)
        self.header = "Service Failure"
        self.event_type = "service"


class PodEntity(Entity):
    """The k8s pod entity"""

    def __init__(self, **kwargs):
        """The k8s pod entity initializer"""
        super().__init__(**kwargs)
        self.header = "Pod Failure"
        self.event_type = "pod"


class VmEntity(Entity):
    """The vm entity"""

    def __init__(self, **kwargs):
        """The k8s pod entity initializer"""
        super().__init__(**kwargs)
        self.header = "Vm Failure"
        self.event_type = "vm"