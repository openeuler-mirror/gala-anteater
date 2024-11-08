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
import time
import json
import os.path
import pprint
import traceback
from typing import List

import numpy as np
import pandas as pd

from anteater.core.slow_node_response import AIJobDetectResult, ResultCode, NodeData
from anteater.core.anomaly import Anomaly, RootCause
from anteater.core.kpi import KPI, ModelConfig, Feature
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.timer import timer
from anteater.utils.log import logger
from anteater.source.metric_loader import MetricLoader
from anteater.model.detector.base import Detector
from anteater.model.process.rank_table_loader import GroupDataLoader
from anteater.model.algorithms.slow_node_algs import time_node_detectors, space_node_detectors


class SlowNodeDetector(Detector):
    def __init__(self, data_loader: MetricLoader, config: ModelConfig, **kwargs):
        """The detector base class initializer"""
        super().__init__(data_loader, **kwargs)
        self.config = config
        self.max_num_normal_results = self.config.params.get("max_num_normal_results", 10)
        self.record_kpi_value = self.config.params.get("record_kpi", False)
        self.hccl_domain, self.rank_table = self._init_hccl_and_rank_table()

    def _init_hccl_and_rank_table(self):
        params = self.config.params
        hccl_domain_path = params.get("hccl_domain_json")
        rank_table_path = params.get("rank_table_json")

        hccl_domain = {}
        rank_table = {}

        if os.path.exists(hccl_domain_path):
            try:
                with open(rank_table_path, 'r', encoding='utf-8') as f_out:
                    hccl_domain = json.load(f_out)
            except Exception:
                logger.error(f"Read hccl domain info fail!")
        if not hccl_domain:
            # 增加手动设置hccl_domain
            hccl_domain = params.get("hccl_domain", {})
        if os.path.exists(rank_table_path):
            try:
                with open(rank_table_path, 'r', encoding='utf-8') as f_out:
                    rank_table = json.load(f_out)
            except Exception:
                logger.error(f"Read rank table info fail!")

        return hccl_domain, rank_table

    @staticmethod
    def npu_id2host_id(machines2devices: dict):
        npu_id2host_id_dict = {}
        npu_ids = []
        hosts_ids = []
        for machine_ip, devices in machines2devices.items():
            if devices == [""]:
                hosts_ids.append(machine_ip)
            else:
                npu_ids.append(machine_ip)

        for npu_id in npu_ids:
            for host_id in hosts_ids:
                if npu_id.split(":")[0] in host_id:
                    npu_id2host_id_dict[npu_id] = host_id
                    break

        return npu_id2host_id_dict, hosts_ids

    def get_host_ids_by_npu_ids(self, npu_ids: dict, npu_id2host_id_dict: dict, hosts_ids: list) -> list:
        host_ids = []
        if npu_ids:
            for npu_id in npu_ids:
                host_id = npu_id2host_id_dict.get(npu_id, "")
                if host_id:
                    host_ids.append(host_id)
        else:
            host_ids = hosts_ids

        return host_ids

    @timer
    def _execute(self, kpis: List[KPI], features: List[Feature], **kwargs) \
            -> List[Anomaly]:
        # save to kafka response
        anomalies = []

        logger.info('Execute cdt model: %s.', self.__class__.__name__)
        start, end = dt.last(minutes=30)
        # 获取machine_ids,
        machines_to_devices = self.get_machines_to_devices(start, end, kpis)
        npu_id2host_id, hosts_ids = self.npu_id2host_id(machines_to_devices)
        group_dataloader = GroupDataLoader(self.hccl_domain, self.rank_table, machines_to_devices)
        group_ranks: list = group_dataloader.get_group_ranks()
        all_results = []
        for kpi in kpis:
            for index, ranks in enumerate(group_ranks):
                logger.info(f"Groups-{index}, metric: {kpi.metric}, start detection.")
                machine_ids: dict = group_dataloader.rank_table_loader.get_group_nodes_by_ranks(ranks)
                host_ids: list = self.get_host_ids_by_npu_ids(machine_ids, npu_id2host_id, hosts_ids)
                group_result = self.group_detect_single_kpi(kpi, machine_ids, host_ids, ranks)
                all_results.extend(group_result)

        response, all_anomaly_nodes = self.gen_final_alarm(kpis, all_results)

        if response.result_code == ResultCode.anomaly:
            all_anomaly_nodes = sorted(list(set(all_anomaly_nodes)))
            anomaly = Anomaly(
                machine_id=json.dumps(all_anomaly_nodes),
                metric="slow_node_metric",
                labels={"instance": "node_ip"},
                score=1.0,
                entity_name="sli",
                details={"detect_method": "slow_node_detection"},
                description=response)
            anomalies.append(anomaly)

        return anomalies

    def gen_final_alarm(self, kpis: List[KPI], detect_results: List):
        response = AIJobDetectResult()
        all_anomaly_nodes = []

        for index, result in enumerate(detect_results):
            try:
                aomaly_devices = result.get("anomaly_devices")
                all_anomaly_nodes.extend(aomaly_devices)
                response = self.group_detect_ret_agg(response, result, kpis)
            except Exception:
                logger.error(traceback.format_exc())
            logger.info("accomplishment: %s/%s", index + 1, len(detect_results))

        return response, all_anomaly_nodes

    def group_detect_single_kpi(self, kpi: KPI, machine_ids: dict, host_ids: list, ranks) -> list:
        """Detects kpi based on signal time series anomaly detection model"""
        # 普罗会一次性抓到所有的数据，需要根据machine_id, device_id去对数据作分组
        metric_name: str = kpi.metric

        all_machines_ts = []
        for machine_id in machine_ids:
            single_machine_ts_list = self.get_kpi_ts_list(metric_name, machine_id, kpi.params)
            if single_machine_ts_list:
                # 根据ranks匹配组内device的指标
                local_ranks = [int(rank) % 8 for rank in ranks]
                for single_machine_ts in single_machine_ts_list:
                    ts_id = int(single_machine_ts.labels.get("id", -1))
                    if ts_id in local_ranks:
                        all_machines_ts.append(single_machine_ts)

        for host_id in host_ids:
            single_machine_ts_list = self.get_kpi_ts_list(metric_name, host_id, kpi.params)
            all_machines_ts.extend(single_machine_ts_list)
        logger.info(f"Metric-{metric_name} single group has data {len(all_machines_ts)}. ranks: {ranks}")

        anomaly_devices = []
        anomaly_locations = {}
        space_anomaly_locations = {}

        detect_data, min_data_len = self.preprocessing_data(metric_name, all_machines_ts)
        detection_results = {
            "anomaly_devices": anomaly_devices,
            "anomaly_locations": anomaly_locations,
            "detect_result_type": "TIME",
            "metric_name": metric_name,
            "group_data": detect_data,
        }
        if min_data_len == 0:
            logger.warning("GROUP data contains EMPTY DATA. GROUP_DATA:%s", pprint.pformat(all_machines_ts))
            return [detection_results]
        logger.info("work on %s, %s start.", metric_name, "slow_node_detection")

        # 时间检测
        # logger.info("work on %s, %s started.", metric_name, "time_node_compare")
        time_anomaly_locations = self.time_node_compare(kpi, detect_data)
        logger.info(f"time_node_compare result: {self.output_anomaly_devices(metric_name, time_anomaly_locations)}.")
        # logger.info("work on %s, %s finished.", metric_name, "time_node_compare")

        # 空间维度对比
        # 若指标空间维度配置为空，则不进行均质化对比
        if kpi.params.get("space_detector") is not None:
            # 四个以上的对象才进行均质化
            if len(all_machines_ts) >= 4:
                # 空间维度对比，输出异常节点
                space_anomaly_locations = self.space_nodes_compare(kpi, detect_data)
                logger.info(
                    f"space_nodes_compare result: {self.output_anomaly_devices(metric_name, space_anomaly_locations)}.")
            else:
                logger.info(f"Skip space nodes compare, due to nodes number {len(all_machines_ts)} is smaller than 4.")
        else:
            logger.info(f"Skip space nodes compare.")

        # 时间空间结果融合
        anomaly_locations, detect_result_type = self.time_space_agg(time_anomaly_locations, space_anomaly_locations,
                                                                    metric_name)

        anomaly_devices = self.output_anomaly_devices(metric_name, anomaly_locations)
        detection_results["anomaly_devices"] = anomaly_devices
        detection_results["anomaly_locations"] = anomaly_locations
        detection_results["detect_result_type"] = detect_result_type

        logger.info(f'''Time and space aggregated result: {anomaly_devices}.''')
        logger.info("work on %s, %s end.\n", metric_name, "slow_node_detection")

        return [detection_results]

    @staticmethod
    def output_anomaly_devices(metric: str, anomaly_location: dict):
        anomaly_devices = []
        for device_info in anomaly_location.keys():
            # 异常点数大于0, 则认为该指标出现异常
            if np.sum(anomaly_location[device_info][metric][1]) > 0:
                anomaly_devices.append(device_info)

        return anomaly_devices

    @staticmethod
    def preprocessing_data(metric_name: str, metric_data: list):
        if len(metric_data) == 0:
            return {}, 0

        detect_data = {}
        length = 0
        for index, metric_ts in enumerate(metric_data):
            time_stamps = metric_ts.time_stamps
            length = len(time_stamps)
            values = metric_ts.values
            labels = metric_ts.labels
            if labels.get("id"):
                device_label = f'''{labels.get("instance")}*{labels.get("id")}'''
            else:
                device_label = f'''{labels.get("instance")}*-1'''
            detect_data[device_label] = pd.DataFrame({"timestamp": time_stamps, metric_name: values})

        return detect_data, length

    def time_node_compare(self, kpi: KPI, detect_data: dict):
        metric_name = kpi.metric
        cfg = kpi.params.get("time_detector", {})
        detector_class = time_node_detectors.get(cfg.get("type"))

        time_node_detector = detector_class(metric_name=metric_name, cfg=cfg)
        time_node_detector.fit(detect_data)
        locations = time_node_detector.predict(detect_data)
        expert_alarm_window_size = kpi.params.get("alarm_filter_window_size")

        for device_info, anomaly_locations in locations.items():
            filter_labels = self.alarm_filter(anomaly_locations[metric_name][1], expert_alarm_window_size)
            locations[device_info][metric_name][1][:] = filter_labels

        return locations

    def space_nodes_compare(self, kpi: KPI, detect_data: dict):
        metric_name = kpi.metric
        cfg = kpi.params.get("space_detector", {})
        detector_class = space_node_detectors.get(cfg.get("type"))
        space_detector = detector_class(cfg)
        df = pd.DataFrame()
        column_list = []
        for device_label, infer_data in detect_data.items():
            df[device_label] = infer_data[metric_name]
            column_list.append(device_label)

        detect_node_data = df[column_list].values
        labels = space_detector.detect(detect_node_data)

        labels = np.swapaxes(labels, 0, 1)
        space_detect_locations = {}

        i = 0
        for device_label in column_list:
            space_detect_locations[device_label] = {}
            space_detect_locations[device_label][metric_name] = detect_data[device_label]["timestamp"], labels[i]
            i += 1
        return space_detect_locations

    def get_kpi_ts_list(self, metric, machine_id: str, kpi_params: dict):
        look_back = self.config.params.get("look_back", 10)
        metric_type = kpi_params.get("metric_type", "device")
        start, end = dt.last(minutes=look_back)

        if metric_type == "device":
            # npu device
            ts_list = self.data_loader.get_metric(start, end, metric, instance=machine_id)
        else:
            # host
            op = kpi_params.get("method", "avg")
            ts_list = self.data_loader.get_metric(start, end, metric, operator=op, keys="instance", instance=machine_id)

        return ts_list

    @staticmethod
    def alarm_filter(labels, alarm_filter_window_size):
        copy_labels = np.zeros(len(labels))
        start_index = alarm_filter_window_size
        alarm_points = set()
        for i in range(start_index, len(labels) + 1):
            is_sequential_alarm = (np.sum(labels[i - alarm_filter_window_size:i]) >= alarm_filter_window_size)
            if not is_sequential_alarm:
                if np.sum(labels[i - alarm_filter_window_size:i]) > 0:
                    alarm_points.add(i - alarm_filter_window_size)
            else:
                copy_labels[i - alarm_filter_window_size:i] = labels[i - alarm_filter_window_size:i]
        # if alarm_points:
        #     logger.info(f"Alert Remove from point loc", list(alarm_points))

        return copy_labels

    @staticmethod
    def time_space_agg(time_anomaly_locations, space_anomaly_locations, metric_name):
        detect_result_type = {}

        for node_id in time_anomaly_locations.keys():
            time_ret = np.sum(time_anomaly_locations[node_id][metric_name][1])
            if space_anomaly_locations:
                space_ret = np.sum(space_anomaly_locations[node_id][metric_name][1])
                # 如果均质化没有报错则消除告警
                # 若空间检测和时间检测结果都为空，则返回正常值
                # 若时间维度和空间维度都出现异常，以空间维度为主返回结果
                if space_ret == 0 or (space_ret > 0 and time_ret >= 0):
                    time_anomaly_locations[node_id][metric_name] = space_anomaly_locations[node_id][metric_name]
                    detect_result_type.setdefault(node_id, {}).setdefault(metric_name, "SPACE")
                else:
                    detect_result_type.setdefault(node_id, {}).setdefault(metric_name, "TIME")
            else:
                detect_result_type.setdefault(node_id, {}).setdefault(metric_name, "TIME")

        return time_anomaly_locations, detect_result_type

    @staticmethod
    def _get_kpi_params(kpis: List[KPI], metric_name):
        for kpi in kpis:
            if kpi.metric == metric_name:
                return kpi.params

        return {}

    def group_detect_ret_agg(self, response, detect_result, kpis: List[KPI]):
        anomaly_device_labels = detect_result.get("anomaly_devices")
        anomaly_locations = detect_result.get("anomaly_locations")
        metric_name = detect_result.get("metric_name")
        detect_result_type = detect_result.get("detect_result_type")
        group_data = detect_result.get("group_data")
        if len(anomaly_device_labels) == 0:
            return response
        else:
            response.result_code = ResultCode.anomaly
        kpi_params = self._get_kpi_params(kpis, metric_name)
        response(kpi_params.get('type', "compute"))

        keep_devices = []
        omitted_devices = []
        for device_label in anomaly_device_labels:
            method_type = detect_result_type.get(device_label, {}).get(metric_name, "TIME")
            if method_type == "SPACE":
                normal_devices = sorted(set(group_data.keys()) - set(anomaly_device_labels))
                keep_devices = normal_devices[:self.max_num_normal_results]
                omitted_devices = normal_devices[self.max_num_normal_results:]
            abnormal_node_data = NodeData(metric_name, device_label, method_type, keep_devices, omitted_devices)
            time_stamp_data, values = anomaly_locations[device_label][metric_name]
            label_dict = dict(zip(time_stamp_data.tolist(), values.tolist()))

            # see user requirements for this real kpi value
            if self.record_kpi_value:
                # record anomaly kpi value
                g_ts, g_value = group_data[device_label].values[:, 0], group_data[device_label].values[:, 1]
                kpi_data = []
                for key, value in sorted(zip(g_ts.tolist(), g_value.tolist()), key=lambda x: x[0]):
                    kpi_data.append({str(key): str(value), "abnormal": label_dict.get(key, 0)})

                abnormal_node_data.kpi_data = kpi_data
            response.abnormal_detail.append(abnormal_node_data)

        if keep_devices:
            for device_label in keep_devices:
                normal_node_data = NodeData(metric_name, device_label, "SPACE")
                # see user requirements for this real kpi value
                if self.record_kpi_value:
                    # record normal kpi data for space compare
                    g_ts, g_value = group_data[device_label].values[:, 0], group_data[device_label].values[:, 1]
                    kpi_data = [{str(key): str(value)} for key, value in zip(g_ts.tolist(), g_value.tolist())]
                    normal_node_data.kpi_data = kpi_data
                response.normal_detail.append(normal_node_data)
        return response
