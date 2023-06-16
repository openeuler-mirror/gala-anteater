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

import json
import logging
import os
from os import makedirs, path
from json import JSONDecodeError
from typing import List

from anteater.core.desc import Description
from anteater.core.kpi import KPI, ModelConfig, Feature, JobConfig
from anteater.utils.constants import ANTEATER_CONFIG_PATH, \
     ANTEATER_MODEL_PATH, ANTEATER_MODULE_PATH


def duplicated_metric(metrics: List[str]):
    if len(metrics) != len(set(metrics)):
        return True

    return False


def load_job_config(file_name) -> JobConfig:
    folder_path = path.realpath(ANTEATER_MODULE_PATH)
    abs_path = path.join(folder_path, file_name)

    with open(abs_path, 'r', encoding='utf-8') as f_out:
        try:
            config = json.load(f_out)
        except JSONDecodeError as e:
            logging.error('JSONDecodeError when parse job file %s',
                          os.path.basename(abs_path))
            raise e

    name = config.get('name')
    job_type = config.get('job_type')
    keywords = config.get('keywords', [])
    root_cause_number = config.get('root_cause_number', 0)

    kpis = [KPI.from_dict(**_conf) for _conf in config.get('KPI')]
    features = [Feature.from_dict(**_conf) for _conf in config.get('Features', [])]

    model_config = None
    if 'OnlineModel' in config:
        name = config['OnlineModel']['name']
        enable = config['OnlineModel']['enable']
        params = config['OnlineModel']['params']
        root_model_path = path.realpath(ANTEATER_MODEL_PATH)
        model_path = path.join(root_model_path, path.splitext(file_name)[0])

        if not path.exists(model_path):
            makedirs(model_path)

        model_config = ModelConfig(name=name, enable=enable,
                                   params=params, model_path=model_path)

    if duplicated_metric([kpi.metric for kpi in kpis]) or \
       duplicated_metric([f.metric for f in features]):
        raise ValueError(f'Existing duplicated metric name'
                         f'in config file: {path.basename(abs_path)}')

    # filter out un-enable kpis
    kpis = [kpi for kpi in kpis if kpi.enable]

    return JobConfig(
        name=name,
        job_type=job_type,
        keywords=keywords,
        root_cause_number=root_cause_number,
        kpis=kpis,
        features=features,
        model_config=model_config
    )


def load_desc(file_name) -> Description:
    """Loads metrics' descriptions"""
    folder_path = path.realpath(ANTEATER_CONFIG_PATH)
    abs_path = path.join(folder_path, file_name)

    with open(abs_path, 'r', encoding='utf-8') as f_out:
        try:
            items = json.load(f_out)
        except JSONDecodeError as e:
            logging.error('JSONDecodeError: when parsing '
                          'file %s', path.basename(abs_path))
            raise e

    desc = Description()
    for item in items:
        metric = item.get('metric', None)
        if not metric:
            raise KeyError('Empty metric name in config file '
                           f'{path.basename(abs_path)}')
        if not metric or \
           desc.in_en(metric) or \
           desc.in_zh(metric):
            raise KeyError(f'Duplicated metric \'{metric}\' in config '
                           f'file {path.basename(abs_path)}')

        desc_en = item.get('en', None)
        desc_zh = item.get('zh', None)

        if not desc_en and not desc_zh:
            raise KeyError(f'Empty en and zh desc on \'{metric}\'')

        if desc_en:
            desc.append_en(metric, desc_en)

        if desc_zh:
            desc.append_zh(metric, desc_zh)

    return desc
