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
Description: The factory of the model initializer.
"""

from anteater.model.algorithms.calibrate import Calibrator, ErrorCalibrator
from anteater.model.algorithms.normalization import Normalization
from anteater.model.algorithms.threshold import Threshold, DynamicThreshold
from anteater.model.algorithms.usad import USADModel
from anteater.model.algorithms.vae import VAEModel


class ModelFactory:
    """The model factory"""

    @staticmethod
    def create_model(name, folder, **kwargs):
        """Create model based on the specific name"""
        if name == 'vae':
            return VAEModel.load(folder, **kwargs.get('vae', {}))

        elif name == 'norm':
            return Normalization.load(folder, **kwargs.get('norm', {}))

        elif name == 'calibrate':
            return Calibrator.load(folder, **kwargs.get('calibrate', {}))

        elif name == 'threshold':
            return Threshold.load(folder, **kwargs.get('threshold', {}))

        elif name == 'dy_threshold':
            return DynamicThreshold.load(folder, **kwargs.get('dy_threshold', {}))

        elif name == 'error_calibrate':
            return ErrorCalibrator.load(folder, **kwargs.get('error_calibrate', {}))

        elif name == 'usad':
            return USADModel.load(folder, **kwargs.get('usad', {}))

        else:
            raise ValueError(f"Unknown model name {name} when model factorization.")
