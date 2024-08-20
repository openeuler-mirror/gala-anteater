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
Description: Some common functions are able to use in this project.
"""

import re
from typing import Union
from datetime import datetime, timedelta

from anteater.utils.log import logger


def divide(x, y):
    """Divide expression in case divide-by-zero problem"""
    if y != 0:
        return x / y
    else:
        return 0


def same_intersection_pairs(first: dict, second: dict):
    """Checks there are same key value pairs between two dictionaries
    intersections by the key
    """
    same_keys = set(first.keys()) & set(second.keys())
    for key in same_keys:
        if first[key] != second[key]:
            return False

    return True


def to_bytes(letter: Union[str, int]) -> int:
    """Converts the string to the number of bytes

    such as:
        - '1k' / '1kb' => 1024
        - '1m' / '1mb' => 1024 * 1024
        - '1g' / '1gb' => 1024 * 1024 * 1024
    """
    size_map = {
        '': 1,
        'b': 1,
        'k': 1024,
        'kb': 1024,
        'm': 1024 * 1024,
        'mb': 1024 * 1024,
        'g': 1024 * 1024 * 1024,
        'gb': 1024 * 1024 * 1024,
    }

    if isinstance(letter, int):
        return letter

    elif isinstance(letter, str):
        try:
            num = int(letter)
        except ValueError:
            num = -1

        if num > 0:
            return num

        letter = letter.lower()
        try:
            num, suffix, _ = re.split('([a-z]+)', letter)
        except ValueError as e:
            logger.error(f'ValueError: parses "{letter}" to the number of bytes!')
            raise e

        if suffix not in size_map:
            raise ValueError(f'Unknown suffix "{suffix}" convert to bytes!')

        return int(num) * size_map.get(suffix)

    else:
        raise ValueError("The type of letter is neither str nor int!")


class GlobalVariable:
    local_data_path = ''
    use_local_data = False
    is_test_model = False
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=15)
    test_end_time = datetime.utcnow()
    test_start_time = test_end_time - timedelta(minutes=5)
    train_end_time = test_end_time - timedelta(minutes=5)
    train_start_time = test_end_time - timedelta(minutes=15)
