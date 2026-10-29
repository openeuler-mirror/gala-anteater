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

import os
import sys
import shutil
import logging
import datetime
from glob import glob

from setuptools import setup, find_packages

# 备份时间戳统一用 UTC，避免 now() 与 strptime() 的时区基准不一致导致 aware/naive 混比
UTC = datetime.timezone.utc
LOGGER = logging.getLogger('gala-anteater-setup')


# 随包分发的文件清单（单一数据源，供清理逻辑与 data_files 共用，避免二者漂移）
CONFIG_FILES = (glob('config/metricinfo.json') +
                glob('config/gala-anteater.yaml') +
                glob('config/log.settings.ini'))
MODULE_FILES = glob('config/module/*')
ENTITY_FILES = glob('config/entity/*')
SERVICE_FILES = glob('service/*')

CFG_PATH = "/etc/gala-anteater"
# 备份文件命名约定：原文件名 + BACKUP_INFIX + 时间戳(BACKUP_TS_FMT)
BACKUP_INFIX = '.bak-'
BACKUP_TS_FMT = '%Y%m%d%H%M%S'
# 备份保留天数，超过此天数的 *.bak-<时间戳> 会被清理，避免堆积；可用环境变量覆盖
try:
    BACKUP_KEEP_DAYS = int(os.environ.get('GALA_ANTEATER_BACKUP_KEEP_DAYS', '7'))
except ValueError:
    BACKUP_KEEP_DAYS = 7


def _backup_and_prune(target_dir, keep_names, timestamp):
    """对 target_dir 下的现存文件先备份，再删除新版本不再分发的陈旧文件。
    - 每个文件都会生成 *.bak-<时间戳>，保证任何改动都可回滚；
    - 仍随包分发的文件保留原文件，交由 data_files 覆盖。
    """
    if not os.path.isdir(target_dir):
        return
    for name in os.listdir(target_dir):
        if BACKUP_INFIX in name:
            # 已是历史备份，跳过：否则会被再备份成嵌套名字（x.bak-T1.bak-T2）
            # 并删除原件，导致时间戳每次被重置、保留期永不命中、
            # 备份数量随安装次数堆积，文件名超过 NAME_MAX 后招 ENAMETOOLONG
            continue
        path = os.path.join(target_dir, name)
        if not os.path.isfile(path):
            continue
        shutil.copy2(path, f'{path}{BACKUP_INFIX}{timestamp}')
        if name not in keep_names:
            # 新版本已不再分发该文件，备份后删除
            os.remove(path)


def _utc_now():
    """统一获取 UTC 当前时间，保证与文件名时间戳同一时区基准。"""
    return datetime.datetime.now(UTC)


def _prune_old_backups(target_dir, keep_days):
    """清理 target_dir 下超过 keep_days 天的 *.bak-<时间戳> 备份，避免长期堆积。
    备份由 shutil.copy2 生成、保留了原文件 mtime，故不能按 mtime 判断，
    需解析文件名内嵌的创建时间戳。keep_days <= 0 时不清理（永久保留）。
    时间戳的生成与解析均按 UTC；旧版以本地时间命名的历史备份会被判为偏新，
    最多延后一个时区偏移量被清理，不影响正确性。
    """
    if not os.path.isdir(target_dir) or keep_days <= 0:
        return
    cutoff = _utc_now() - datetime.timedelta(days=keep_days)
    for name in os.listdir(target_dir):
        if BACKUP_INFIX not in name:
            continue
        ts_str = name.rsplit(BACKUP_INFIX, 1)[-1]
        try:
            parsed = datetime.datetime.strptime(ts_str, BACKUP_TS_FMT)
            ts = parsed.replace(tzinfo=UTC)
        except ValueError:
            continue
        if ts < cutoff:
            try:
                os.remove(os.path.join(target_dir, name))
            except OSError:
                pass


def cleanup_old_config():
    """安装前安全清理旧版本配置。
    仅备份并清理 config/module/entity 三个受管子目录，不再无差别删除整个
    /etc/gala-anteater，避免误删用户已调好的生产配置及运行时生成的文件。
    先按保留天数清理过期备份，再为本次要覆盖/删除的文件生成新备份。
    无权限或目录不存在时静默跳过，交由 data_files 写入阶段给出明确报错。
    """
    if not os.path.isdir(CFG_PATH):
        return
    timestamp = _utc_now().strftime(BACKUP_TS_FMT)
    managed = {
        'config': {os.path.basename(p) for p in CONFIG_FILES},
        'module': {os.path.basename(p) for p in MODULE_FILES},
        'entity': {os.path.basename(p) for p in ENTITY_FILES},
    }
    try:
        for subdir, keep in managed.items():
            sub_dir = os.path.join(CFG_PATH, subdir)
            _prune_old_backups(sub_dir, BACKUP_KEEP_DAYS)
            _backup_and_prune(sub_dir, keep, timestamp)
    except OSError as e:
        LOGGER.warning('skip config cleanup: %s', e)


# 打包/构建场景的安装参数：命中任一即说明安装目标被重定向到临时根目录
PACKAGING_OPT_PREFIXES = ('--root', '--prefix', '--install-root')
PACKAGING_EXACT_OPTS = ('--skip-build',)


def _is_packaging_install(argv):
    """判断本次 install 是否被重定向到 root/prefix（RPM 打包的 buildroot、
    pip 的临时安装目录等）。此类场景下安装目标不是宿主机真实目录，而
    cleanup 操作的路径是硬编码的绝对路径，必须跳过，避免构建过程读写、
    篡改宿主机 /etc，污染可复现构建。
    """
    for arg in argv:
        hit_prefix = arg.startswith(PACKAGING_OPT_PREFIXES)
        hit_exact = arg in PACKAGING_EXACT_OPTS
        if hit_prefix or hit_exact:
            return True

    return False


# 仅在执行 install/develop 时清理，避免 --help / 打包 / pip 安装元数据等场景误删系统文件；
# 带 --root/--prefix 的打包安装（如 RPM %{buildroot}）同样跳过，防止篡改构建机宿主机 /etc
if any(cmd in sys.argv for cmd in ('install', 'develop')) and \
        not _is_packaging_install(sys.argv):
    cleanup_old_config()

setup(
    name="gala_anteater",
    version="2.0.1",
    author="Zhenxing Li",
    author_email="lizhenxing11@huawei.com",
    description="Times Series Anomaly Detection Platform on Operating System",
    url="https://atomgit.com/openeuler/gala-anteater",
    keywords=["Anomaly Detection", "Time Series Analysis", "Operating System"],
    packages=find_packages(where=".", exclude=("tests", "tests.*")),
    data_files=[
        ('/etc/gala-anteater/config/', CONFIG_FILES),
        ('/etc/gala-anteater/module/', MODULE_FILES),
        ('/etc/gala-anteater/entity/', ENTITY_FILES),
        ('/usr/lib/systemd/system/', SERVICE_FILES),
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
        "statsmodels"
    ],
    entry_points={
        "console_scripts": [
            "gala-anteater=anteater.main:main",
        ]
    }
)
