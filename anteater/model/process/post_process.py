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

from typing import Dict

import numpy as np
import pandas as pd

from anteater.model.algorithms.loss import gan_loss
from anteater.model.algorithms.spot import Spot


class PostProcessor:
    """The post processor for deep learning model output"""

    def __init__(self, params: Dict):
        """The """
        self.params = params.get('postprocessor')
        self.score_type = self.params.get('score_type')
        self.alpha = self.params.get('alpha')
        self.q = self.params.get('q')
        self.level = self.params.get('level')

        self.spot = None

    @staticmethod
    def get_anomalies(x, anomalies):
        """Gets the results predicted by the model"""
        anomalies = np.array(anomalies)
        pred_anomaly_times = []
        for i in anomalies:
            pred_anomaly_times.append(x[i])

        result_df = pd.DataFrame({"timestamp": x.index})
        pred_index = result_df[result_df["timestamp"].isin(pred_anomaly_times)].index
        result_df.loc[pred_index, "pred"] = 1

        return result_df["pred"]

    def fit(self, scores):
        """The post-processor fit and initialize local variables"""
        self.spot = Spot(self.q)
        self.spot.initialize(scores[:1000], level=self.level)
        self.spot.run(scores[1000:])

    def compute_score(self, x, x_g, x_g_d):
        """Calculates the anomaly score by reconstruction error"""
        recon_x = gan_loss(x, x_g, x_g_d, self.score_type, self.alpha)

        scores = np.sum(recon_x, axis=1)
        diff_scores = np.diff(scores)
        diff_scores_0 = diff_scores[0]
        return np.insert(diff_scores, 0, diff_scores_0)

    def spot_run(self, scores):
        """Obtains dynamic threshold for anomaly scores by the SPOT algorithm"""

        result = self.spot.run(scores)
        return result.get('thresholds')
