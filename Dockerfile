# syntax=docker/dockerfile:1

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
# Create: 2022-06-01
#
# Dockerfile for building openEuler aops gala-anteater docker image.
#
# Usage:
#       docker build -f Dockerfile -t gala-anteater:1.0.0 .
#       docker run -v /etc/gala-anteater:/etc/gala-anteater -it gala-anteater:1.0.0
#

# x86 - base image
FROM python:3.7-slim-buster

# ARM -base image
# FROM python:3.7

ENV PYTHONPATH "${PYTHONPATH}:/home/gala-anteather/anteater"

WORKDIR /home/gala-anteather

COPY requirements.txt requirements.txt
COPY . /home/gala-anteather

COPY config/gala-anteater.yaml ./config/
COPY config/module/ ./config/module/

COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh

# Setting the pip3 source
# RUN pip3 config set global.index-url https://mirrors.tools.huawei.com/pypi/simple \
#     && pip3 config set install.trusted-host mirrors.tools.huawei.com

RUN pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "/entrypoint.sh" ]

CMD ["python3", "./anteater/main.py"]