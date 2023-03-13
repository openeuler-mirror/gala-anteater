import copy
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from os import path, sep
from typing import Dict, List

from sklearn.metrics import classification_report

from anteater.utils.common import divide
from anteater.utils.log import logger


@dataclass
class Fault:
    fault_type: datetime = None
    injection_time: str = None
    duration: int = None
    start: datetime = None
    end: datetime = None

    def contains(self, timestamp):
        return self.start <= timestamp <= self.end + timedelta(minutes=2)


def load_labels() -> Dict[str, List[Fault]]:
    folder_path = path.dirname(path.dirname(path.realpath(__file__)))
    label_folder = path.join(folder_path, sep.join(['data', 'label']))

    faults = defaultdict(list)
    for file_name in os.listdir(label_folder):
        sub_file = os.path.join(label_folder, file_name)
        if path.isfile(sub_file):
            idx = file_name.rfind('.')
            machine = file_name[:idx]

            with open(sub_file, 'r') as f:
                reader = csv.reader(f)
                is_header = True
                for row in reader:
                    if is_header:
                        is_header = False
                        continue

                    fault = Fault()
                    fault.fault_type = row[0]
                    fault.injection_time = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S").astimezone()
                    fault.duration = int(row[2])
                    fault.start = datetime.fromtimestamp(int(row[5])).astimezone()
                    fault.end = datetime.fromtimestamp(int(row[6])).astimezone()

                    faults[machine].append(fault)

    return faults


def adjust_pred(y_true: List[int], y_pred: List[int]):
    if len(y_true) != len(y_pred):
        raise ValueError("'y_true' and 'y_pred' must be the same size")

    adjusted_y_pred = copy.deepcopy(y_pred)
    anomaly_state = False
    for i, _ in enumerate(adjusted_y_pred):
        if y_true[i] and adjusted_y_pred[i]:
            anomaly_state = True
            for j in range(i, 0, -1):
                if not y_true[j]:
                    break
                else:
                    if not adjusted_y_pred[j]:
                        adjusted_y_pred[j] = 1

        elif not y_true[i]:
            anomaly_state = False

        if anomaly_state:
            adjusted_y_pred[i] = 1

    return adjusted_y_pred


class TestEvaluation:
    def __init__(self, anomalies: List, labels: Dict[str, List[Fault]]):
        self.anomalies = anomalies
        self.labels = labels

    def evaluate(self, start: datetime, end: datetime):
        machine_ids = self.labels.keys()

        for machine_id in machine_ids:
            labels = self.labels[machine_id]
            anomalies = []
            for anomaly in self.anomalies:
                anomaly = json.loads(anomaly)
                if anomaly['Attributes']['entity_id'].startswith(machine_id):
                    anomalies.append(anomaly)

            curr = start
            total_points = [curr]
            while curr <= end:
                curr = curr + timedelta(minutes=1)
                total_points.append(curr)

            y_true = [0 for _ in total_points]
            y_pred = [0 for _ in total_points]

            for i, point in enumerate(total_points):
                for fault in labels:
                    if fault.contains(point):
                        y_true[i] = 1

                for anomaly in anomalies:
                    stamp = divide(int(anomaly['Timestamp']), 10**3)
                    timestamp = datetime.fromtimestamp(stamp).astimezone()
                    if timestamp - timedelta(seconds=30) <= point <= timestamp + timedelta(seconds=30):
                        y_pred[i] = 1

            y_pred = adjust_pred(y_true, y_pred)
            logger.info(f'Evaluation: machine {machine_id} performance: ')
            rpt = classification_report(y_true, y_pred)

            logger.info(rpt)
