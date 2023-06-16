#!/usr/bin/python3
# ******************************************************************************
# Copyright (c) 2023 Huawei Technologies Co., Ltd.
# gala-anteater is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# ******************************************************************************/


class Description:
    """The metric description class
    Which contains en or zh languages for each metrics
    """

    file_name = "descriptions.json"

    def __init__(self) -> None:
        """The description class initializer"""
        self.metric2en = {}
        self.metric2zh = {}

    def get_en(self, metric: str) -> str:
        """Gets english description"""
        return self.metric2en.get(metric, '')

    def get_zh(self, metric: str) -> str:
        """Gets chinese description"""
        return self.metric2zh.get(metric, '')

    def append_en(self, metric: str, des: str) -> None:
        """Appends metric and en- desc pair"""
        self.metric2en[metric] = des

    def append_zh(self, metric: str, des: str) -> None:
        """Appends metric and zh- desc pair"""
        self.metric2zh[metric] = des

    def in_en(self, metric) -> bool:
        """Whether metric is already in en desc or not"""
        return metric in self.metric2en

    def in_zh(self, metric) -> bool:
        """Whether metric is already in zh desc or not"""
        return metric in self.metric2zh
