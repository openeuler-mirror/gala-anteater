#!/usr/bin/python3
# ******************************************************************************
# Copyright (c) 2023 Huawei Technologies Co., Ltd.
# gala-anteater is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# ******************************************************************************/


# CONFIG PATH
ANTEATER_CONFIG_PATH = '/etc/gala-anteater/config'
ANTEATER_MODULE_PATH = '/etc/gala-anteater/module'
ANTEATER_MODEL_PATH = '/etc/gala-anteater/models'

# COMMON TIME STEMP
POINTS_MINUTE = 12
POINTS_HOUR = 12 * 60
POINTS_DAY = 12 * 60 * 24

# PARAMS
THRESHOLD = 'threshold'
LOOK_BACK = 'look_back'

# LABEL NAME
MACHINE_ID = 'machine_id'
TGID = 'tgid'
PID = 'pid'
COMM = 'comm'
IP = 'ip'
SERVER_IP = 'server_ip'
CONTAINER_ID = 'container_id'
POD_NAME = 'pod_name'
DEVICE = 'device'
DEV_NAME = 'dev_name'
DISK_NAME = 'disk_name'
FSNAME = 'Fsname'

# JVM LABEL NAME
GC = 'gc'
POOL = 'pool'
AREA = 'area'
AREA_HEAP = 'heap'
AREA_NON_HEAP = 'nonheap'
OLD_G_COLLECTORS = [
    'MarkSweepCompact',
    'PS MarkSweep',
    'ConcurrentMarkSweep',
    'G1 Mixed Generation',
    'G1 Old Generation',
    'G1 Concurrent GC'
]
PS_OLD_G = 'PS Old Gen|G1 Old Gen|Tenured Gen'
METASPACE = 'Metaspace'
