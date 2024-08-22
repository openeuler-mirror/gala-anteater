import re
import os
import argparse
from dataclasses import dataclass


@dataclass
class RootCauseInfo:
    time_stamp: str
    top_n: str
    root_machine: str
    root_metric: str
    ano_score: float
    fv_pod: str


def split_by_keywords(text, keywords):
    pattern = '|'.join(map(re.escape, keywords))
    parts = re.split(pattern, text)
    return parts


# parts = split_by_keywords("2023-11-27 17:28:30,347 - root - INFO - cause_infer.py:285 - timestamp: 1701077211236, top1, root_cause_metric: gala_gopher_l7_latency_sum*6256854a-da25-485b-a505-2b6452f2df81, rw_score: 0.1588", [',', '*'])
# parts = split_by_keywords("timestamp: 1703320843936, top1, fv_pod:dcf142e5-230e-43a7-9cfc-db899de5e3f4root_cause_metric:gala_gopher_endpoint_tcp_retran_syn*2b9f846a-a9f4-42c2-b7c3-d8d67edb579f, rw_score:0.0608", [':', ',', '*'])
#
# print(parts)

def parse_top_root(parts):
    time_stamp = parts[1].strip()
    top_seq = parts[2].strip()
    fv_pod = parts[4].strip()
    root_machine = parts[7].strip()
    root_metric = parts[6].strip()
    ano_score = float(parts[-1].strip())

    root_cause_info = RootCauseInfo(time_stamp=time_stamp, top_n=top_seq, root_machine=root_machine,
                                    root_metric=root_metric, ano_score=ano_score, fv_pod=fv_pod)
    print(root_cause_info)

    return root_cause_info


def insert_top_result(fv_info, right, top_key="top1"):
    if top_key in fv_info.keys():
        fv_info[top_key] += right
    else:
        fv_info[top_key] = right

    return fv_info


def cal_tp(cal_data, pod_name='2b9f846a-a9f4-42c2-b7c3-d8d67edb579f'):
    final_results = dict()
    for data in cal_data:
        tmp_result = dict()
        fv_pod = data.fv_pod
        top_n = data.top_n
        tmp_result[fv_pod] = dict()

        if data.root_machine == pod_name:
            right = 1
        else:
            right = 0
        tmp_result[fv_pod] = insert_top_result(tmp_result[fv_pod], right, top_n)

        if fv_pod not in final_results.keys():
            final_results.update(tmp_result)
        else:
            if top_n not in final_results[fv_pod].keys():
                final_results[fv_pod].update(tmp_result[fv_pod])
            else:
                final_results[fv_pod][top_n] += tmp_result[fv_pod][top_n]

    return final_results


def parse_single_data(_file_path):
    with open(_file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    _valid_data = []
    for line in data:
        for index in range(3):
            result_key = f'top{index + 1}, fv_pod'
            if result_key in line:
                line_infos = split_by_keywords(line, [':', ',', '*'])
                root_cause_info = parse_top_root(line_infos)
                _valid_data.append(root_cause_info)
                break

    return _valid_data


def update_test_tops(test_results, single_result):
    top1_flag = single_result.get('top1', 0) > 0
    top2_flag = single_result.get('top2', 0) > 0
    top3_flag = single_result.get('top3', 0) > 0

    if top1_flag:
        test_results['top1'] += 1

    if top1_flag or top2_flag:
        test_results['top2'] += 1

    if top1_flag > 0 or top2_flag or top3_flag > 0:
        test_results['top3'] += 1

    return test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default='./log/backend4_r8_data80/')
    parser.add_argument("--true_pod", type=str, default='2b9f846a-a9f4-42c2-b7c3-d8d67edb579f')
    parser.add_argument('--test', action='store_true', help="test set flag for top3 acc.")
    args = parser.parse_args()

    root_path = args.root_path
    true_pod = args.true_pod
    test_set_results = {
        "top1": 0,
        "top2": 0,
        "top3": 0,
        "test_num": 0,
    }
    for file in os.listdir(root_path):
        file_path = os.path.join(root_path, file)
        valid_data = parse_single_data(file_path)
        results = cal_tp(valid_data, true_pod=true_pod)
        test_set_results['test_num'] += len(results.keys())
        for result in results.values():
            test_set_results = update_test_tops(test_set_results, result)
    print(test_set_results)
    for i in range(3):
        key = f"top{str(i + 1)}"
        test_set_results[key] = test_set_results[key] / test_set_results["test_num"]

    print(test_set_results)
