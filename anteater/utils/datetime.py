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

from datetime import datetime, timezone, timedelta


class DateTimeManager:
    """The datetime manager which could freeze or unfreeze datetime
    for getting a consistency time on the model.
    """

    __freeze = False
    __freeze_utc_now = None

    @classmethod
    def utc_now(cls) -> datetime:
        """Gets the current utc time"""
        if cls.__freeze:
            return cls.__freeze_utc_now
        else:
            return datetime.now(timezone.utc).astimezone()

    @classmethod
    def update_and_freeze(cls):
        """Updates and freezes current datetime"""
        cls.__freeze = True
        cls.__freeze_utc_now = datetime.now(timezone.utc).astimezone()

    @classmethod
    def unfreeze(cls):
        """Unfreezes the datetime"""
        cls.__freeze = False

    @classmethod
    def last(cls, seconds=0, minutes=0, hours=0):
        """Gets the latest datetime range based on current date"""
        return ((cls.utc_now() -
                timedelta(seconds=seconds, minutes=minutes, hours=hours)),
                cls.utc_now())
