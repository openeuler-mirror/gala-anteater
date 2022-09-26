# -*- coding: utf-8 -*-
from glob import glob

from setuptools import setup, find_packages

setup(
    name="gala_anteater",
    version="0.0.2",
    author="Zhenxing Li",
    author_email="lizhenxing11@huawei.com",
    description="Times Series Anomaly Detection Platform on Operating System",
    url="https://gitee.com/openeuler/A-Ops/tree/master/gala-anteater",
    keywords=["Anomaly Detection", "Time Series Analysis", "Operating System"],
    packages=find_packages(where="."),
    data_files=[
        ('/etc/gala-anteater/config', glob('anteater/config/*')),
        ('/etc/gala-anteater/models', glob('anteater/models/*')),
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
            "gala-anteater = anteater.main:main",
        ]
    }
)
