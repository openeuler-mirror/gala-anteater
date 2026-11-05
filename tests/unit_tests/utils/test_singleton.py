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
"""Unit tests for anteater.utils.singleton."""

import threading
import unittest


class SingletonStub:
    """A plain class whose instances record construction arguments."""

    def __init__(self, value=None):
        self.value = value


class TestSingleton(unittest.TestCase):
    """Tests for the Singleton metaclass."""

    def _make_singleton_class(self):
        from anteater.utils.singleton import Singleton

        class SingletonClass(SingletonStub, metaclass=Singleton):
            pass

        return SingletonClass

    def test_singleton_should_return_same_instance(self):
        """Two constructions return the identical object"""
        singleton_class = self._make_singleton_class()

        first = singleton_class("first")
        second = singleton_class("second")

        self.assertIs(first, second)
        # the first construction wins, later arguments are ignored
        self.assertEqual(first.value, "first")

    def test_singleton_should_be_independent_per_class(self):
        """Two singleton classes do not share their instances"""
        first_class = self._make_singleton_class()
        second_class = self._make_singleton_class()

        first = first_class("a")
        second = second_class("b")

        self.assertIsNot(first, second)
        self.assertEqual(first.value, "a")
        self.assertEqual(second.value, "b")

    def test_singleton_should_be_thread_safe(self):
        """Concurrent constructions yield exactly one instance"""
        singleton_class = self._make_singleton_class()
        barrier = threading.Barrier(8)
        instances = []

        def construct():
            barrier.wait()
            instances.append(singleton_class("t"))

        threads = [threading.Thread(target=construct) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(instances), 8)
        for instance in instances:
            self.assertIs(instance, instances[0])


if __name__ == "__main__":
    unittest.main()
