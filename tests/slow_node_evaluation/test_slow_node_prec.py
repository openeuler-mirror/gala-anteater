# coding=utf-8
"""
Copyright (c) Huawei Technologies Co., Ltd. 2020-2028. All rights reserved.
Description:
FileName：test_slow_node_prec.py
Author: h00568282/huangbin 
Create Date: 2024/10/31 20:00
Notes:

"""
import os
import argparse
from datetime import datetime, timedelta, timezone

from anteater.config import AnteaterConf
from anteater.utils.constants import ANTEATER_CONFIG_PATH
from anteater.provider.kafka import KafkaProvider
from anteater.utils.log import logger

# (3362911 3362919 3363053 3363115 3363139 3363209 3363285 3363452)
npu2tgid = {
    0: 3362911,
    1: 3362919,
    2: 3363053,
    3: 3363115,
    4: 3363139,
    5: 3363209,
    6: 3363285,
    7: 3363452,
}


def init_config() -> AnteaterConf:
    """initialize anteater config"""
    conf = AnteaterConf()
    conf.load_from_yaml(ANTEATER_CONFIG_PATH)

    return conf


class TestSlowNodePrec():
    def __init__(self):
        conf = init_config()
        self.kafka_provider = KafkaProvider(conf.kafka)

    def get_anomaly_data_from_kafka(self, delta_hour: float = 1) -> list:
        # logger.info(f'Run slow node evaluation: {self.__class__.__name__}!')
        now_time = datetime.now(timezone.utc).astimezone()
        try:
            anomaly_datas = self.kafka_provider.range_query(now_time - timedelta(hours=delta_hour), now_time)
        except Exception:
            import traceback
            logger.info(f"{traceback.format_exc()}")
            logger.error(f'Get from kafka error!')

            anomaly_datas = []

        logger.info(f"Get anomaly data {len(anomaly_datas)} from kafka")

        return anomaly_datas

    @staticmethod
    def extract_anomaly_nodes(kafka_anomaly_datas: list) -> dict:
        detected_anomaly = []
        for kafka_anomaly_data in kafka_anomaly_datas:
            time_stamp = int(kafka_anomaly_data["Timestamp"] / 1000)  # s
            kafka_anomaly_info_list = kafka_anomaly_data["Attributes"]["abnormalDetail"]

            anomaly_info = {"time_stamp": time_stamp}

            for kafka_anomaly_info in kafka_anomaly_info_list:
                node_ip = kafka_anomaly_info["serverIp"]
                device_id = int(kafka_anomaly_info["objectId"])
                if device_id == -1:
                    anomaly_info[node_ip] = []
                    continue
                if node_ip not in anomaly_info.keys():
                    anomaly_info[node_ip] = [device_id]
                else:
                    if device_id not in anomaly_info[node_ip]:
                        anomaly_info[node_ip].append(device_id)

            detected_anomaly.append(anomaly_info)

        return detected_anomaly

    @staticmethod
    def extract_inject_times(log_file: str, start: str = "start", end: str = "end"):
        inject_info_list = []
        with open(log_file, "r") as f:
            lines = f.readlines()

        for idx in range(len(lines) - 1):
            line = lines[idx].strip()
            next_line = lines[idx + 1].strip()
            if start in line and next_line.isdigit():
                device_id = line.split("device_id:")[1].split(",")[0]
                inject_index = line.split("/")[0]
                start_time = int(next_line)
                inject_info = {"index": inject_index,
                               "device_id": device_id,
                               "start_time": start_time}
            elif end in next_line and line.isdigit():
                end_time = int(line)
                time_delta = end_time - start_time
                inject_info["end_time"] = start_time + 2 * time_delta
                inject_info_list.append(inject_info)

        return inject_info_list

    def eval(self, inject_log="/home/workspace/hbdir/log_data/inject.log", delta_hours: float = 1):
        kafka_anomaly_data_list = self.get_anomaly_data_from_kafka(delta_hours)
        anomaly_data_list = self.extract_anomaly_nodes(kafka_anomaly_data_list)
        logger.info(f"anomaly_data_list: {anomaly_data_list}")

        inject_info_list = self.extract_inject_times(inject_log)
        logger.info(f"inject_info_list: {inject_info_list}")

        inject_num = len(inject_info_list)
        detect_num = 0
        all_nodes_prec = {}
        for inject_info in inject_info_list:
            detected = False
            start_time = int(inject_info.get("start_time", -1))
            end_time = int(inject_info.get("end_time", -1))
            device_id = int(inject_info.get("device_id", -10))

            if start_time == -1 or end_time == -1:
                # 记录不完整，临时中断，则跳过注入故障
                continue
            if device_id not in all_nodes_prec.keys():
                all_nodes_prec[device_id] = {"gt": 1}
            else:
                all_nodes_prec[device_id]["gt"] += 1
            for anomaly_data in anomaly_data_list:
                anomaly_time = int(anomaly_data["time_stamp"])
                if start_time < anomaly_time <= end_time:
                    logger.info(f"Match time, start: {start_time}, anomaly:{anomaly_time}, end: {end_time}.")
                    for anomaly_node, anomaly_device_ids in anomaly_data.items():
                        if anomaly_node == "time_stamp":
                            continue
                        if device_id in anomaly_device_ids:
                            detected = True
                            logger.info(
                                f"Match deviceid, inject id: {device_id}, anomaly ids:{anomaly_device_ids}.")
                            detect_num += 1
                            if "tp" not in all_nodes_prec[device_id]:
                                all_nodes_prec[device_id]["tp"] = 1
                            else:
                                all_nodes_prec[device_id]["tp"] += 1
                            break
                # 当检测窗口20min持续覆盖故障注入时间段，会持续上报告警
                # 故障注入段内仅最多匹配1次检测结果
                if detected:
                    break

        if inject_num:
            prec = detect_num / inject_num
            for device_id, cal_info in all_nodes_prec.items():
                if "tp" in cal_info.keys():
                    cal_info["prec"] = cal_info["tp"] / cal_info["gt"]
                else:
                    cal_info["prec"] = 0.
                    cal_info["tp"] = 0

            logger.info(f"Detect_num/inject_num: {detect_num}/{inject_num}, Slow node detection prec is {prec}")
            logger.info(f"All node prec: {all_nodes_prec}")
        else:
            logger.info(f"inject num is {inject_num}. Please check inject log.")

    @staticmethod
    def check_ai_model_running_status():
        output = {}
        f = os.popen('/npu-smi info')
        npu_info = f.read()
        npu_info_str = ""
        for _chr in npu_info:
            npu_info_str += _chr

        split_str = "+===========================+===============+====================================================+"
        npu_info = npu_info_str.split(split_str)
        for index in range(10, 18):
            if "No running processes found in NPU" in npu_info[index]:
                continue
            device_info = npu_info[index].strip().split("|")
            logger.info(f"device_info: {device_info},")
            device_id = device_info[1].split("       ")[0].strip()
            device_tgid = device_info[2].strip()
            output[device_id] = device_tgid
        logger.info(f"output: {output},")

        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SLOW NODE DETECTION')
    parser.add_argument('--inject_log', type=str, help='inject log file path.')
    parser.add_argument('--delta_hours', type=float, help='anomaly detection time far from now time.')
    args = parser.parse_args()
    inject_log_path = args.inject_log
    delta_hours_param = args.delta_hours

    logger.info("Start slow node detection evaluation.")
    validator = TestSlowNodePrec()
    validator.eval(inject_log=inject_log_path, delta_hours=delta_hours_param)
    logger.info("Finish slow node detection evaluation.")
    # validator.check_ai_model_running_status()

    exit()
