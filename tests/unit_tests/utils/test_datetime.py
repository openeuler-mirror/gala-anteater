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
"""Unit tests for anteater.utils.datetime."""

import unittest
from datetime import datetime, timedelta

import pytz

from anteater.utils.datetime import DateTime, DateTimeManager


class TestDateTime(unittest.TestCase):
    """Tests for the DateTime singleton."""

    def setUp(self):
        # DateTime is a Singleton: drop the shared instance before every
        # test so each case starts from a clean state
        DateTime._instance = None

    def test_datetime_should_now_correct_from_string(self):
        """A str timestamp is parsed with the documented format"""
        date_time = DateTime("2022-09-01 08:00:00")

        self.assertEqual(
            date_time.now(),
            datetime.strptime("2022-09-01 08:00:00", "%Y-%m-%d %H:%M:%S").astimezone(),
        )

    def test_datetime_should_now_correct_from_int(self):
        """An int timestamp is interpreted as a POSIX timestamp"""
        date_time = DateTime(1662000000)

        self.assertEqual(
            date_time.now(),
            datetime.fromtimestamp(1662000000).astimezone(),
        )

    def test_datetime_should_now_correct_from_datetime(self):
        """A datetime object is returned unchanged"""
        stamp = datetime(2022, 9, 1, 8, 0, 0, tzinfo=pytz.utc)
        date_time = DateTime(stamp)

        self.assertEqual(date_time.now(), stamp)

    def test_datetime_should_now_correct_when_none(self):
        """None falls back to the current UTC time"""
        date_time = DateTime(None)

        before = datetime.utcnow().replace(tzinfo=pytz.utc) - timedelta(seconds=5)
        now = date_time.now()
        after = datetime.utcnow().replace(tzinfo=pytz.utc) + timedelta(seconds=5)

        self.assertTrue(before <= now <= after)

    def test_datetime_should_raise_on_unsupported_type(self):
        """Unsupported timestamp types raise a TypeError"""
        date_time = DateTime(3.14)

        with self.assertRaises(TypeError):
            date_time.now()

    def test_datetime_should_update_and_keep_singleton(self):
        """update() changes the frozen time and the class stays a singleton"""
        date_time = DateTime("2022-09-01 08:00:00")
        other = DateTime()

        self.assertIs(date_time, other)

        date_time.update("2022-09-02 09:30:00")

        self.assertEqual(
            other.now(),
            datetime.strptime("2022-09-02 09:30:00", "%Y-%m-%d %H:%M:%S").astimezone(),
        )


class TestDateTimeManager(unittest.TestCase):
    """Tests for the DateTimeManager freeze/unfreeze helpers."""

    def setUp(self):
        DateTime._instance = None
        DateTimeManager.unfreeze()

    def test_utc_now_should_return_current_time_when_unfrozen(self):
        """Without freezing, utc_now() tracks the wall clock"""
        before = datetime.utcnow().replace(tzinfo=pytz.utc) - timedelta(seconds=5)
        now = DateTimeManager.utc_now()
        after = datetime.utcnow().replace(tzinfo=pytz.utc) + timedelta(seconds=5)

        self.assertTrue(before <= now <= after)

    def test_utc_now_should_return_frozen_time_when_frozen(self):
        """After update_and_freeze(), utc_now() keeps the frozen value"""
        DateTimeManager.update_and_freeze()
        frozen = DateTimeManager.utc_now()

        self.assertEqual(DateTimeManager.utc_now(), frozen)

    def test_last_should_return_range_ended_at_now(self):
        """last() returns a (start, end) pair with the requested offset"""
        DateTimeManager.update_and_freeze()
        frozen = DateTimeManager.utc_now()

        start, end = DateTimeManager.last(minutes=15)

        self.assertEqual(end, frozen)
        self.assertEqual(start, frozen - timedelta(minutes=15))

    def test_unfreeze_should_restore_current_time(self):
        """unfreeze() makes utc_now() track the wall clock again"""
        DateTimeManager.update_and_freeze()
        frozen = DateTimeManager.utc_now()
        DateTimeManager.unfreeze()

        before = datetime.utcnow().replace(tzinfo=pytz.utc) - timedelta(seconds=5)
        now = DateTimeManager.utc_now()
        after = datetime.utcnow().replace(tzinfo=pytz.utc) + timedelta(seconds=5)

        self.assertNotEqual(now, frozen)
        self.assertTrue(before <= now <= after)


if __name__ == "__main__":
    unittest.main()
