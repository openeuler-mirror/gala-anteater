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

import json
import codecs
from os import path
import os
import stat
from typing import List
import libconf


META_FOLDER = '/gala-gopher/src/probes/'
METRICINFO_FILE = 'metricinfo_tmp.json'
SKIPPED_TABLE = ['example', 'event']
SKIPPED_TYPE = ['key', 'label', 'Unknown']


def parse_meta(meta_file) -> List:
    """Parses the meta file and get metricinfos"""
    meta_file = path.realpath(meta_file)
    metricinfo = []
    with open(meta_file, 'r', encoding='utf-8') as reader:
        config = libconf.load(reader)

        for item in config['measurements']:
            entity_name = item['entity_name']
            if entity_name in SKIPPED_TABLE:
                continue

            for field in item['fields']:
                if field['type'] in SKIPPED_TYPE:
                    continue

                metricinfo.append({
                    'metric': f'gala_gopher_{entity_name}_{field["name"]}',
                    'entity_name': entity_name,
                    'type': field['type'],
                    'en': field['description']})

    return metricinfo


def load_metricinfo():
    """Loads metricinfo from the folder"""
    metricinfos = []

    for dirpath, _, filenames in os.walk(META_FOLDER):
        for filename in filenames:
            if not filename.endswith('.meta'):
                continue

            filepath = os.path.join(dirpath, filename)
            metricinfos.extend(parse_meta(filepath))

    metricinfos.sort(key=lambda x: x['metric'])
    metrics = set()

    dedup_metricinfos = []
    for item in metricinfos:
        if item['metric'] not in metrics:
            dedup_metricinfos.append(item)
            metrics.add(item['metric'])

    modes = stat.S_IWUSR | stat.S_IRUSR
    flags = os.O_WRONLY | os.O_CREAT
    with os.fdopen(os.open(METRICINFO_FILE, flags, modes), "w") as writer:
        json.dump(dedup_metricinfos, writer, indent=2)


def update_metricinfo():
    """Update metricinfo from old one"""
    metricinfo = {}
    with codecs.open(METRICINFO_FILE, 'r', encoding='utf-8') as reader:
        data = json.load(reader)
        for item in data:
            metricinfo[item['metric']] = item

    old_file = 'metricinfo.json'
    old_info = {}
    with codecs.open(old_file, 'r', encoding='utf-8') as reader:
        data = json.load(reader)
        for item in data:
            old_info[item['metric']] = item

    for metric, item in old_info.items():
        if metric not in metricinfo:
            raise ValueError(f'Unknown metric name {metric}')

    new_info = []
    for metric, item in metricinfo.items():
        if metric in old_info and old_info[metric]['zh']:
            item['zh'] = old_info[metric]['zh']

        new_info.append(item)

    new_file = 'metricinfo_new.json'
    with codecs.open(new_file, 'w', encoding='utf-8') as writer:
        json.dump(new_info, writer, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    load_metricinfo()
    # update_metricinfo()
