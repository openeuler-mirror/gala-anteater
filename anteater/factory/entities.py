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

from anteater.source.entity import Entity, AppEntity, ServiceEntity,\
    PodEntity, VmEntity

ENTITIES = {
    'app': AppEntity,
    'service': ServiceEntity,
    'pod': PodEntity,
    'vm': VmEntity,
}


class EntityFactory:
    """The template factory"""

    @staticmethod
    def get_entity(name: str, **kwargs) -> Entity:
        """Gets template by name"""
        if name not in ENTITIES:
            raise KeyError(f'Unknown entity name \'{name}\'')

        return ENTITIES[name](**kwargs)
