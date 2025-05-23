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
import os 

# 安装前清理旧版本配置文件
cfg_path = "/etc/gala-anteater"
for root, dirs, files in os.walk(cfg_path):
    for file in files:
        os.remove(os.path.join(root, file))

ser = "/usr/lib/systemd/system/gala-anteater.service"
if os.path.isfile(ser):
    os.remove(ser)

setup(
    name="gala_anteater",
    version="2.0.1",
    author="Zhenxing Li",
    author_email="lizhenxing11@huawei.com",
    description="Times Series Anomaly Detection Platform on Operating System",
    url="https://gitee.com/openeuler/gala-anteater",
    keywords=["Anomaly Detection", "Time Series Analysis", "Operating System"],
    packages=find_packages(where=".", exclude=("tests", "tests.*")),
    data_files=[
        ('/etc/gala-anteater/config/', glob('config/metricinfo.json')),
        ('/etc/gala-anteater/config/', glob('config/gala-anteater.yaml')),
        ('/etc/gala-anteater/config/', glob('config/log.settings.ini')),
        ('/etc/gala-anteater/module/', glob('config/module/*')),
        ('/etc/gala-anteater/entity/', glob('config/entity/*')),
        ('/usr/lib/systemd/system/', glob('service/*')),
    ],
    install_requires=[
        "APScheduler",
        "kafka-python",
        "joblib",
        "numpy",
        "pandas",
        "requests",
        "scikit_learn",
        "scipy",
        "torch",
        "networkx",
        "pyArango",
        "pingouin",
        "statsmodels"
    ],
    entry_points={
        "console_scripts": [
            "gala-anteater=anteater.main:main",
        ]
    }
)
