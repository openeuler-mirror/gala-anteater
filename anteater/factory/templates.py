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

from anteater.source.template import Template, AppAnomalyTemplate, \
    JVMAnomalyTemplate, SysAnomalyTemplate, SimpleAnomalyTemplate, NetAnomalyTemplate, SlowNodeTemplate

TEMPLATES = {
    'app': AppAnomalyTemplate,
    'net': NetAnomalyTemplate,
    'sys': SysAnomalyTemplate,
    'jvm': JVMAnomalyTemplate,
    'simple': SimpleAnomalyTemplate,
    "slow_node": SlowNodeTemplate
}


class TemplateFactory:
    """The template factory"""

    @staticmethod
    def get_template(name: str, **kwargs) -> Template:
        """Gets template by name"""
        if name not in TEMPLATES:
            raise KeyError(f'Unknown template name \'{name}\'')

        return TEMPLATES[name](**kwargs)
