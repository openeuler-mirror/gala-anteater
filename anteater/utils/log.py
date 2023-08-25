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
Description: the implementation of logging
"""

import os
import logging.config
import logging
from os.path import realpath, join

LOG_CONFIG_PATH = "/etc/gala-anteater/config"
LOG_DATA_PATH = "/var/log/gala-anteater/"


class Log:
    """The wrapped logging module"""

    __flag = None

    def __new__(cls, *args, **kwargs):
        if not cls.__flag:
            cls.__flag = super().__new__(cls)
        return cls.__flag

    def __init__(self):
        log_config_file = join(realpath(LOG_CONFIG_PATH), "log.settings.ini")
        if not os.path.isfile(log_config_file):
            raise FileNotFoundError("log.settings.ini not found!")  # pylint:disable=undefined-variable

        log_dir = realpath(LOG_DATA_PATH)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_filename = os.path.join(log_dir, 'gala-anteater.log')
        logging.config.fileConfig(log_config_file, defaults={'filename': log_filename})
        self.logger = logging.getLogger('anteater')

    def get_logger(self):
        """Gets the wrapped logger object"""
        return self.logger


logger = Log().get_logger()
