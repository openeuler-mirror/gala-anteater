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
from os import makedirs, path, sep
from json import JSONDecodeError
from typing import List

from anteater.core.kpi import KPI, ModelConfig, Feature, JobConfig
from anteater.utils.log import logger

ANTEATER_MODULE_PATH = '/etc/gala-anteater/config/module'
ANTEATER_MODEL_PATH = '/etc/gala-anteater/models'


def anteater_model_path():
    """Gets the model folder for model serialization"""
    root_path = os.path.realpath(ANTEATER_MODEL_PATH)

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    return root_path


def load_metric_operator():
    """Loads metric name and corresponding operator"""
    folder_path = path.dirname(path.dirname(path.realpath(__file__)))
    metrics_file = path.join(folder_path, sep.join(["observe", "metrics.csv"]))

    logger.info(f"Loads metric and operators from file: {metrics_file}")

    metric_operators = []
    with open(metrics_file, 'r', encoding='utf-8') as f:
        for line in f:
            metric_name, operator = line.strip().split(",")
            metric_operators.append((metric_name, operator))

    return metric_operators


def load_metric_description():
    """Loads metric name and it's descriptions"""
    folder_path = path.dirname(path.dirname(path.realpath(__file__)))
    metrics_file = path.join(folder_path, sep.join(["observe", "description.csv"]))

    logger.info(f"Loads metric and descriptions from file: {metrics_file}")

    descriptions = {}
    with open(metrics_file, 'r', encoding='utf-8') as f:
        for line in f:
            name, dsp = line.strip().split(",")
            descriptions[name] = dsp

    return descriptions


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
            logger.error(f'JSONDecodeError when parse job'
                         f'file {os.path.basename(abs_path)}')
            raise e

    name = config['name']
    job_type = config['job_type']
    root_cause_number = config['root_cause_number']
    kpis = [KPI(**_conf) for _conf in config['KPI']]
    features = [Feature(**_conf) for _conf in config['Features']]

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
        root_cause_number=root_cause_number,
        kpis=kpis,
        features=features,
        model_config=model_config
    )
