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
"""Unit tests for anteater.utils.timer."""

import logging
import sys
import unittest
from types import ModuleType
from unittest import mock

# anteater.utils.timer imports anteater.utils.log, whose module level
# `logger = Log().get_logger()` reads /etc/gala-anteater/config and raises
# FileNotFoundError on a machine without a deployed environment. Stub the
# log module BEFORE importing timer so the unit tests run standalone.
if 'anteater.utils.log' not in sys.modules:
    _fake_log = ModuleType('anteater.utils.log')
    _fake_log.logger = logging.getLogger('anteater')
    sys.modules['anteater.utils.log'] = _fake_log

from anteater.utils.log import logger
from anteater.utils.timer import timer


class TestTimer(unittest.TestCase):
    """Tests for the timer decorator."""

    def test_timer_should_return_wrapped_result(self):
        """The decorator forwards the wrapped function's return value"""
        @timer
        def add(a, b):
            return a + b

        self.assertEqual(add(1, 2), 3)

    def test_timer_should_forward_arguments_and_preserve_name(self):
        """functools.wraps keeps the wrapped function's metadata"""
        @timer
        def sample(x):
            return x * 2

        self.assertEqual(sample.__name__, "sample")
        self.assertEqual(sample(21), 42)

    def test_timer_should_log_class_name_for_methods(self):
        """On a method call the log line carries the receiver class name"""
        class Worker:
            @timer
            def run(self, value):
                return value

        worker = Worker()
        with mock.patch.object(logger, "info") as mock_info:
            result = worker.run(7)

        self.assertEqual(result, 7)
        mock_info.assert_called_once()
        format_args = mock_info.call_args.args
        self.assertEqual(format_args[0], "%s.%s costs %.2f seconds!")
        self.assertEqual(format_args[1], "Worker")
        self.assertEqual(format_args[2], "run")

    def test_timer_should_propagate_exceptions_without_logging(self):
        """A raised exception propagates and is not logged as finished"""
        @timer
        def failing():
            raise RuntimeError("boom")

        with mock.patch.object(logger, "info") as mock_info:
            with self.assertRaises(RuntimeError):
                failing()

        mock_info.assert_not_called()

    def test_timer_should_log_a_non_negative_elapsed_time(self):
        """The elapsed time logged is a non-negative number.

        Note: the decorator reads args[0] to build the log line, so the
        wrapped call must carry at least one positional argument.
        """
        @timer
        def noop(_placeholder):
            return None

        with mock.patch.object(logger, "info") as mock_info:
            noop(None)

        elapsed = mock_info.call_args.args[3]
        self.assertGreaterEqual(elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()
