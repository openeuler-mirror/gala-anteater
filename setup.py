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

from glob import glob

from setuptools import setup, find_packages

setup(
    name="gala_anteater",
    version="1.1.0",
    author="Zhenxing Li",
    author_email="lizhenxing11@huawei.com",
    description="Times Series Anomaly Detection Platform on Operating System",
    url="https://gitee.com/openeuler/A-Ops/tree/master/gala-anteater",
    keywords=["Anomaly Detection", "Time Series Analysis", "Operating System"],
    packages=find_packages(where=".", exclude=("tests", "tests.*")),
    data_files=[
        ('/etc/gala-anteater/config/', glob('config/metricinfo.json')),
        ('/etc/gala-anteater/config/', glob('config/gala-anteater.yaml')),
        ('/etc/gala-anteater/config/', glob('config/log.settings.ini')),
        ('/etc/gala-anteater/module/', glob('config/module/*')),
        ('/usr/lib/systemd/system/', glob('service/*')),
    ],
    install_requires=[
        "APScheduler",
        "kafka-python>=2.0.2",
        "joblib",
        "numpy",
        "pandas",
        "requests",
        "scikit_learn",
        "scipy",
        "torch"
    ],
    entry_points={
        "console_scripts": [
            "gala-anteater=anteater.main:main",
        ]
    }
)
