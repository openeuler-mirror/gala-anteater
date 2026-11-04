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
"""Unit tests for anteater.utils.common."""

import logging
import sys
import unittest
from datetime import datetime, timedelta
from types import ModuleType

# anteater.utils.common imports anteater.utils.log, whose module level
# `logger = Log().get_logger()` reads /etc/gala-anteater/config and raises
# FileNotFoundError on a machine without a deployed environment. Stub the
# log module BEFORE importing common so the unit tests run standalone.
if 'anteater.utils.log' not in sys.modules:
    _fake_log = ModuleType('anteater.utils.log')
    _fake_log.logger = logging.getLogger('anteater')
    sys.modules['anteater.utils.log'] = _fake_log

from anteater.utils.common import divide, GlobalVariable, \
    same_intersection_pairs, to_bytes


class TestCommon(unittest.TestCase):

    def test_to_bytes(self):
        # given
        letter_expects = {
            123456: 123456,
            '123456': 123456,
            '64b': 64,
            '64k': 64 * 1024,
            '64kb': 64 * 1024,
            '64m': 64 * 1024 * 1024,
            '64mb': 64 * 1024 * 1024,
            '1024mb': 1024 * 1024 * 1024,
            '64g': 64 * 1024 * 1024 * 1024,
            '64gb': 64 * 1024 * 1024 * 1024,
            '64K': 64 * 1024,
            '64KB': 64 * 1024,
            '64M': 64 * 1024 * 1024,
            '64MB': 64 * 1024 * 1024,
            '64G': 64 * 1024 * 1024 * 1024,
            '64GB': 64 * 1024 * 1024 * 1024,
        }

        # then
        for letter, expect in letter_expects.items():
            result = to_bytes(letter)
            self.assertEqual(result, expect)

    def test_to_bytes_should_raise_on_unknown_suffix(self):
        """An unknown unit suffix raises a ValueError"""
        with self.assertRaises(ValueError):
            to_bytes('64x')

    def test_to_bytes_should_raise_on_other_types(self):
        """Types other than str/int raise a ValueError"""
        with self.assertRaises(ValueError):
            to_bytes(3.14)

    def test_divide_should_return_quotient(self):
        """Non-zero divisor returns the real quotient"""
        self.assertEqual(divide(6, 3), 2.0)
        self.assertEqual(divide(7, 2), 3.5)
        self.assertEqual(divide(-6, 3), -2.0)

    def test_divide_should_return_zero_on_zero_divisor(self):
        """A zero divisor returns 0 instead of raising"""
        self.assertEqual(divide(6, 0), 0)
        self.assertEqual(divide(0, 0), 0)

    def test_same_intersection_pairs_should_pass_on_equal_pairs(self):
        """Equal values on the shared keys return True"""
        self.assertTrue(same_intersection_pairs({'a': 1, 'b': 2}, {'a': 1}))
        self.assertTrue(same_intersection_pairs({}, {'a': 1}))
        self.assertTrue(same_intersection_pairs({'a': 1}, {'b': 1}))

    def test_same_intersection_pairs_should_fail_on_different_pairs(self):
        """Different values on a shared key return False"""
        self.assertFalse(same_intersection_pairs({'a': 1, 'b': 2}, {'a': 2}))

    def test_global_variable_should_default_time_window(self):
        """The default time window is 15 minutes with a 5 minute test span"""
        now = datetime.utcnow()

        self.assertEqual(
            GlobalVariable.end_time - GlobalVariable.start_time,
            timedelta(minutes=15),
        )
        self.assertEqual(
            GlobalVariable.test_end_time - GlobalVariable.test_start_time,
            timedelta(minutes=5),
        )
        self.assertEqual(
            GlobalVariable.train_end_time - GlobalVariable.train_start_time,
            timedelta(minutes=10),
        )
        # the defaults are computed at import time, so they are recent
        self.assertLessEqual(now - GlobalVariable.end_time, timedelta(minutes=5))


if __name__ == "__main__":
    unittest.main()
