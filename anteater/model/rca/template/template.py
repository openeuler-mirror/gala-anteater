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

from abc import abstractmethod


class Template:
    """The app anomaly template"""
    def __init__(self, timestamp, metric, entity_name):
        self.timestamp = timestamp
        self.metric = metric
        self.entity_name = entity_name

        self.score = 0
        self.labels = {}
        self.entity_id = ""
        self.description = ""
        self.cause_metrics = []
        self.keywords = []

    @abstractmethod
    def get_template(self):
        pass
