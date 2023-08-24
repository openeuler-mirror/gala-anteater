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
import os
from os import makedirs, path
from json import JSONDecodeError
from typing import List

from anteater.core.desc import Description
from anteater.core.kpi import KPI, ModelConfig, Feature, JobConfig
from anteater.utils.constants import ANTEATER_CONFIG_PATH, \
     ANTEATER_MODEL_PATH, ANTEATER_MODULE_PATH
from anteater.utils.log import logger


def validate_dup(metrics: List[str]):
    """Validates whether existing duplicated metrics or not"""
    if len(metrics) != len(set(metrics)):
        return True

    return False


def load_job_config(filepath) -> JobConfig:
    """Loads job config from the file"""
    with open(filepath, 'r', encoding='utf-8') as f_out:
        try:
            config = json.load(f_out)
        except JSONDecodeError as e:
            logger.error('JSONDecodeError when parse job file %s',
                         path.basename(filepath))
            raise e

    job_name = config.get('name')
    enable = config.get('enable', False)
    detector = config.get('detector')
    template = config.get('template')
    keywords = config.get('keywords', [])
    root_cause_num = config.get('root_cause_num', 0)

    kpis = [KPI.from_dict(**_conf) for _conf in config.get('kpis')]
    features = [Feature.from_dict(**_conf) for _conf in config.get('features', [])]

    model_config = None
    if 'model_config' in config:
        name = config['model_config']['name']
        params = config['model_config']['params']
        root_model_path = path.realpath(ANTEATER_MODEL_PATH)
        model_path = path.join(root_model_path, path.basename(filepath))

        if not path.exists(model_path):
            makedirs(model_path)

        model_config = ModelConfig(name=name, params=params, model_path=model_path)

    if validate_dup([kpi.metric for kpi in kpis]) or \
       validate_dup([f.metric for f in features]):
        raise ValueError(f'Existing duplicated metric name'
                         f'in config file: {path.basename(filepath)}')

    # filter out un-enable kpis
    kpis = [kpi for kpi in kpis if kpi.enable]

    return JobConfig(
        name=job_name,
        enable=enable,
        detector=detector,
        template=template,
        keywords=keywords,
        root_cause_num=root_cause_num,
        kpis=kpis,
        features=features,
        model_config=model_config
    )


def load_jobs():
    """Loads all jobs configs from module path"""
    folder = path.realpath(ANTEATER_MODULE_PATH)
    filenames = os.listdir(folder)
    for name in filenames:
        filepath = os.path.join(folder, name)
        if not path.isfile(filepath) or not name.endswith('.json'):
            continue

        yield load_job_config(filepath)


def load_desc(file_name) -> Description:
    """Loads metrics' descriptions"""
    folder_path = path.realpath(ANTEATER_CONFIG_PATH)
    abs_path = path.join(folder_path, file_name)

    with open(abs_path, 'r', encoding='utf-8') as f_out:
        try:
            items = json.load(f_out)
        except JSONDecodeError as e:
            logger.error('JSONDecodeError: when parsing file %s',
                         path.basename(abs_path))
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
