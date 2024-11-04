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

from anteater.model.detector.base import Detector
from anteater.model.detector.jvm_oom_detector import JVMOOMDetector
from anteater.model.detector.n_sigma_detector import NSigmaDetector
from anteater.model.detector.tcp_establish_n_sigma_detector import \
    TcpEstablishNSigmaDetector
from anteater.model.detector.tcp_trans_latency_n_sigma_detector import \
    TcpTransLatencyNSigmaDetector
from anteater.model.detector.th_base_detector import ThBaseDetector
from anteater.model.detector.usad_detector import UsadDetector
from anteater.model.detector.vae_detector import VAEDetector
from anteater.model.detector.disruption_detector import ContainerDisruptionDetector
from anteater.model.detector.slow_node_detector import SlowNodeDetector


DETECTORS = {
    'n-sigma': NSigmaDetector,
    'n-sigma-tcp-establish': TcpEstablishNSigmaDetector,
    'n-sigma-tcp-latency': TcpTransLatencyNSigmaDetector,
    'th-base': ThBaseDetector,
    'usad': UsadDetector,
    'vae': VAEDetector,
    'jvm': JVMOOMDetector,
    'container-disruption': ContainerDisruptionDetector,
    'slow-node-detection': SlowNodeDetector
}


class DetectorFactory:
    """The detector factory"""

    @staticmethod
    def get_detector(name: str, data_loader, **kwargs) -> Detector:
        """Gets detector by name"""
        if name not in DETECTORS:
            raise KeyError(f'Unknown detector name \'{name}\'')

        return DETECTORS[name](data_loader, **kwargs)
