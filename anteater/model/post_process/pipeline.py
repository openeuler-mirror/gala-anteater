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
"""
Time:
Author:
Description: The post-process pipeline contains some basic post-processes
"""

from typing import Iterable, List

from anteater.model.post_process.base import PostProcess


class PostProcessPipe(PostProcess):
    """The post process pipeline"""

    def __init__(self, process: Iterable[PostProcess]):
        """The post process pipeline initializer"""
        self.processes = list(process)

    def __call__(self, scores: List[float]):
        """The callable object"""
        for process in self.processes:
            scores = process(scores)

        return scores

    def train(self, scores):
        """Trains the each elements in the pipeline"""
        for process in self.processes:
            scores = process.train(scores)

        return scores

    def error_rate(self, look_back):
        """Computes the error rate of the last element in the pipeline"""
        return self.processes[-1].error_rate(look_back)
