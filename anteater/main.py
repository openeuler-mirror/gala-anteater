#!/usr/bin/python3
# ******************************************************************************
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
# licensed under the Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#     http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
# PURPOSE.
# See the Mulan PSL v2 for more details.
# ******************************************************************************/
"""
Time:
Author:
Description: The main function of gala-anteater project.
"""

import os
from datetime import datetime, timezone, timedelta
from functools import partial

from apscheduler.schedulers.blocking import BlockingScheduler

from anteater.config import AnteaterConfig
from anteater.model.hybrid_model import HybridModel
from anteater.model.post_model import PostModel
from anteater.model.sli_model import SLIModel
from anteater.service.kafka import EntityVariable
from anteater.utils.common import update_metadata, sent_to_kafka, get_kafka_message
from anteater.utils.log import Log

ANTEATER_DATA_PATH = os.environ.get('ANTEATER_DATA_PATH') or "/etc/gala-anteater/"

log = Log().get_logger()


def init_config() -> AnteaterConfig:
    """initialize anteater config"""
    config = AnteaterConfig()
    config.load_from_yaml(ANTEATER_DATA_PATH)

    return config


def anomaly_detection(hybrid_model: HybridModel, sli_model: SLIModel, post_model: PostModel, config: AnteaterConfig):
    """Run anomaly detection model periodically"""
    utc_now = datetime.now(timezone.utc).astimezone()

    if not EntityVariable.variable:
        log.info("Configuration hasn't been updated.")
        return

    log.info(f"START: run anomaly detection model.")

    machine_ids, dfs = hybrid_model.get_inference_data(utc_now)

    for machine_id, df in zip(machine_ids, dfs):
        if df.shape[0] == 0:
            log.warning(f"Not data was founded for the target machine {machine_id}, please check it first!")

        y_pred = hybrid_model.predict(df)
        hybrid_abnormal = hybrid_model.is_abnormal(y_pred)
        sli_anomalies = sli_model.detect(utc_now, machine_id)

        if sli_anomalies:
            rec_anomalies = post_model.top_n_anomalies(utc_now, machine_id, top_n=60)

            for anomalies in sli_anomalies:
                msg = get_kafka_message(utc_now, y_pred.tolist(), machine_id, anomalies, rec_anomalies)
                sent_to_kafka(msg, config)

            log.info(f"END: abnormal events were detected on machine: {machine_id}.")
        else:
            log.info(f"END: no abnormal events on machine {machine_id}.")

    hybrid_model.online_training(utc_now)


def main():
    log.info("Load gala-anteater conf")
    utc_now = datetime.now(timezone.utc).astimezone()

    config = init_config()

    sub_thread = update_metadata(config)

    hybrid_model = HybridModel(config)
    sli_model = SLIModel(config)
    post_model = PostModel(config)

    if not hybrid_model.model or config.hybrid_model.retrain:
        log.info("Start to re-train the model based on last day metrics dataset!")
        end_time = utc_now - timedelta(hours=config.hybrid_model.look_back)
        x = hybrid_model.get_training_data(end_time, utc_now)
        if x.empty:
            log.error("Error")
        else:
            log.info(f"The shape of training data: {x.shape}")
            hybrid_model.training(x)

    log.info(f"Schedule recurrent job with time interval {config.schedule.duration} minute(s).")
    scheduler = BlockingScheduler()
    scheduler.add_job(partial(anomaly_detection, hybrid_model, sli_model, post_model, config),
                      trigger="interval", minutes=config.schedule.duration)
    scheduler.start()

    sub_thread.join()


if __name__ == '__main__':
    main()
