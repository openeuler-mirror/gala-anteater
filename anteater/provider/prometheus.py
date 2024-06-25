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
Description: The implementation of PrometheusAdapter client to fetch time series data.
"""

from anteater.config import PrometheusConf
from anteater.provider.base import TimeSeriesProvider


class PrometheusAdapter(TimeSeriesProvider):
    """The PrometheusAdapter client to consume time series data"""

    def __init__(self, server, port, step):
        """The PrometheusAdapter client initializer"""
        self.query_url = f"http://{server}:{port}/api/v1/query_range"
        super().__init__(self.query_url, int(step))

    def get_headers(self):
        """Gets the requests headers of prometheus"""
        return {}


def load_prometheus_client(config: PrometheusConf) -> PrometheusAdapter:
    """Load and initialize the prometheus client"""
    server = config.server
    port = config.port
    step = config.steps
    client = PrometheusAdapter(server, port, step)

    return client
