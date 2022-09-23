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

import argparse
import os
from datetime import datetime, timezone, timedelta
from functools import partial

from apscheduler.schedulers.blocking import BlockingScheduler

from anteater.config import AnteaterConfig
from anteater.model.hybrid_model import HybridModel
from anteater.model.post_model import PostModel
from anteater.model.sli_model import SLIModel
from anteater.service.kafka import EntityVariable
from anteater.utils.common import update_metadata, sent_to_kafka, get_kafka_message, update_config
from anteater.utils.log import Log

log = Log().get_logger()

ANTEATER_DATA_PATH = "/etc/gala-anteater/"


def init_config() -> AnteaterConfig:
    """initial anteater config"""
    data_path = os.environ.get('ANTEATER_DATA_PATH') or ANTEATER_DATA_PATH
    conf = AnteaterConfig.load_from_yaml(data_path)

    return conf


def str2bool(arg):
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def arg_parser():
    parser = argparse.ArgumentParser(
        description="gala-anteater project for operation system KPI metrics anomaly detection")

    parser.add_argument("-ks", "--kafka_server",
                        help="The kafka server ip", type=str, required=True)
    parser.add_argument("-kp", "--kafka_port",
                        help="The kafka server port", type=str,  required=True)
    parser.add_argument("-ps", "--prometheus_server",
                        help="The prometheus server ip", type=str, required=True)
    parser.add_argument("-pp", "--prometheus_port",
                        help="The prometheus server port", type=str, required=True)
    parser.add_argument("-m", "--model",
                        help="The machine learning model - random_forest, vae",
                        type=str, default="vae", required=False)
    parser.add_argument("-d", "--duration",
                        help="The time interval of scheduling anomaly detection task (minutes)",
                        type=int, default=1, required=False)
    parser.add_argument("-r", "--retrain",
                        help="If retrain the vae model or not",
                        type=str2bool, nargs='?', const=True, default=False, required=False)
    parser.add_argument("-l", "--look_back",
                        help="Look back window for model training (days)",
                        type=int, default=4, required=False)
    parser.add_argument("-t", "--threshold",
                        help="The model threshold (0, 1), the bigger value, the more strict of anomaly",
                        type=float, default=0.8, required=False)
    parser.add_argument("-sli", "--sli_time",
                        help="The sli time threshold", type=int, default=400, required=False)
    arguments = vars(parser.parse_args())

    return arguments


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
        sli_anomalies, default_anomalies = sli_model.detect(utc_now, machine_id)

        if sli_anomalies or hybrid_abnormal:
            rec_anomalies = post_model.top_n_anomalies(utc_now, machine_id, top_n=60)

            if not sli_anomalies:
                sli_anomalies = default_anomalies

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
    parser = arg_parser()
    config = update_config(config, parser)

    sub_thread = update_metadata(config)

    hybrid_model = HybridModel(config)
    sli_model = SLIModel(config)
    post_model = PostModel(config)

    if not hybrid_model.model or parser["retrain"]:
        log.info("Start to re-train the model based on last day metrics dataset!")
        end_time = utc_now - timedelta(days=parser["look_back"])
        x = hybrid_model.get_training_data(end_time, utc_now)
        if x.empty:
            log.error("Error")
        else:
            log.info(f"The shape of training data: {x.shape}")
            hybrid_model.training(x)

    log.info(f"Schedule recurrent job with time interval {parser['duration']} minute(s).")
    scheduler = BlockingScheduler()
    scheduler.add_job(partial(anomaly_detection, hybrid_model, sli_model, post_model, config),
                      trigger="interval", minutes=parser["duration"])
    scheduler.start()

    sub_thread.join()


if __name__ == '__main__':
    main()
