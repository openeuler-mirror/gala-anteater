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

from anteater.config import AnteaterConf
from anteater.provider.aom import create_aom_collector
from anteater.provider.base import TimeSeriesProvider
from anteater.provider.prometheus import load_prometheus_client


CLIENTS = {
    'prometheus': load_prometheus_client,
    'aom': create_aom_collector
}


class DataClientFactory:
    """The data collector factory"""

    @staticmethod
    def get_client(name: str, config: AnteaterConf) -> TimeSeriesProvider:
        """Gets data collector based on the data source name"""
        if name == 'prometheus':
            return CLIENTS[name](config.prometheus)
        elif name == 'aom':
            return CLIENTS[name](config.aom)
        else:
            raise ValueError(f'Unknown data source name: \'{name}\'.')
