import json
import os
from os import path, sep
from json import JSONDecodeError
from typing import List, Tuple

from anteater.core.feature import Feature
from anteater.core.kpi import KPI
from anteater.utils.log import logger


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


def load_kpi_feature(file_name) -> Tuple[List[KPI], List[Feature]]:
    folder_path = path.dirname(path.dirname(path.dirname(path.realpath(__file__))))
    abs_path = path.join(folder_path, 'config', 'module', file_name)

    with open(abs_path, 'r', encoding='utf-8') as f_out:
        try:
            params = json.load(f_out)
        except JSONDecodeError as e:
            logger.error(f'JSONDecodeError when parse job'
                         f'file {os.path.basename(abs_path)}')
            raise e

    kpis = [KPI(**param) for param in params.get('KPI')]
    features = [Feature(**param) for param in params.get('Features')]

    if duplicated_metric([kpi.metric for kpi in kpis]) or \
       duplicated_metric([f.metric for f in features]):
        raise ValueError(f'Existing duplicated metric name'
                         f'in config file: {os.path.basename(abs_path)}')

    return kpis, features
