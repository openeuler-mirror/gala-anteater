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


def divide(x, y):
    """Divide expression in case divide-by-zero problem"""
    if y != 0:
        return x / y
    else:
        return 0


def same_intersection_key_value(first: dict, second: dict):
    """Checks there are same key value pairs between two dictionaries
    intersections by the key
    """
    same_keys = set(first.keys()) & set(second.keys())
    for key in same_keys:
        if first[key] != second[key]:
            return False

    return True
