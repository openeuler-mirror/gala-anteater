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
Time: 2025-01-17
Author: wangfl
Description: The main function of gala-anteater project.
"""
import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.ERROR) 


def read_table_data_pandas(file_path):
    """
    此函数使用 pandas 从文件读取表格数据。

    参数:
    file_path (str): CSV 文件的路径。

    返回:
    pandas.DataFrame: 包含表格数据的数据框。
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        logging.error(f"Error: The file {file_path} does not exist.")
        return None
    except Exception as e:
        logging.error(f"Error: An error occurred while reading the file {file_path}: {e}")
        return None


def calculate_thresholds(window):
    """
    计算窗口内的上下阈值。

    参数:
    window (pandas.DataFrame): 滑动窗口的数据。

    返回:
    tuple: 包含上下阈值的元组。
    """
    mean_value = window['value'].mean()
    std_value = window['value'].std()
    threshold = 3 * std_value
    upper_threshold = mean_value + threshold
    lower_threshold = mean_value - threshold
    return upper_threshold, lower_threshold


def check_deviation(window, upper_threshold, lower_threshold):
    """
    检查窗口内的数据是否偏离阈值。

    参数:
    window (pandas.DataFrame): 滑动窗口的数据。
    upper_threshold (float): 上阈值。
    lower_threshold (float): 下阈值。

    返回:
    pandas.Series: 偏离状态的布尔序列。
    """
    return ((window['value'] > upper_threshold) | (window['value'] < lower_threshold))


def check_consecutive_deviation(deviation_status_list, consecutive_count=5):
    """
    检查连续偏离点是否超过指定数量。

    参数:
    deviation_status_list (list): 偏离状态列表。
    consecutive_count (int): 连续偏离点的阈值，默认为 5。

    返回:
    list: 标记为故障点的布尔列表。
    """
    corrected_status = []
    consecutive_count_current = 0
    for status in deviation_status_list:
        if status:
            consecutive_count_current += 1
        else:
            consecutive_count_current = 0
        if consecutive_count_current >= consecutive_count:
            corrected_status.append(True)
        else:
            corrected_status.append(False)
    return corrected_status


def analyze_table_data_pandas(df, window_size=600):
    """
    此函数使用 pandas 检查 value 是否偏离均值分布 3 倍以上，
    并将结果保存在新列 deviation_status 中。
    进一步检查滑窗内连续 5 个以上的偏离点，将其标记为故障点，其余为正常点。
    滑窗间不重叠。

    参数:
    df (pandas.DataFrame): 包含表格数据的数据框。
    window_size (int): 滑窗大小，默认为 600。

    返回:
    pandas.DataFrame: 包含 'timestamp', 'value', 'deviation_status' 列的数据框。
    """
    df['deviation_status'] = False
    df['lower_threshold'] = np.nan
    df['upper_threshold'] = np.nan
    num_windows = len(df) // window_size
    for i in range(num_windows):
        start = i * window_size
        end = (i + 1) * window_size
        window = df.iloc[start:end]
        upper_threshold, lower_threshold = calculate_thresholds(window)
        deviation_status = check_deviation(window, upper_threshold, lower_threshold)
        deviation_status_list = deviation_status.tolist()
        corrected_status = check_consecutive_deviation(deviation_status_list)
        df.loc[start:end - 1, 'deviation_status'] = corrected_status
        df.loc[start:end - 1, 'lower_threshold'] = lower_threshold
        df.loc[start:end - 1, 'upper_threshold'] = upper_threshold
    return df


def plot_table_data(df, output_path, window_size):
    """
    此函数使用 matplotlib 绘制表格数据，并标记出故障点，同时增加方格线，保存图像，并绘制上下阈值线。

    参数:
    df (pandas.DataFrame): 包含 'timestamp', 'value', 'deviation_status' 列的数据框。
    output_path (str): 图像保存的路径。
    window_size (int): 滑窗大小。
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    # 绘制正常点
    normal_df = df[~df['deviation_status']]
    ax.plot(normal_df['timestamp'], normal_df['value'], label='Normal', marker='o', linestyle='-', color='blue')
    # 绘制故障点
    fault_df = df[df['deviation_status']]
    ax.plot(fault_df['timestamp'], fault_df['value'], label='Fault', marker='x', linestyle='None', color='red',
            markersize=10)
    # 绘制上下阈值线
    ax.plot(df['timestamp'], df["upper_threshold"], label='Upper Threshold', linestyle='--', color='green')
    ax.plot(df['timestamp'], df["lower_threshold"], label='Lower Threshold', linestyle='--', color='orange')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Value')
    ax.set_title('Table Data Analysis')
    ax.legend()
    # 添加方格线
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.savefig(output_path)
    plt.close(fig)


def save_analysis_result(df, result_path):
    """
    此函数将分析结果保存到 CSV 文件中。

    参数:
    df (pandas.DataFrame): 包含分析结果的数据框。
    result_path (str): 结果保存的路径。
    """
    try:
        df.to_csv(result_path, index=False)
    except Exception as e:
        logging.error(f"Error: An error occurred while saving the file {result_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Table Data Analysis')
    parser.add_argument('directory', type=str, nargs='?', default='./test',
                        help='Path to the directory containing CSV files')
    parser.add_argument('--window_size', type=int, default=600, help='Window size for analysis')
    args = parser.parse_args()

    # 遍历目录下的所有文件
    for root, dirs, files in os.walk(args.directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                # 获取输入文件的基本名称（不包含路径）
                base_name = os.path.basename(file_path)
                file_name = os.path.splitext(base_name)[0]
                output_image_path = os.path.join(root, f'{file_name}.png')
                output_result_path = os.path.join(root, f'{file_name}_result.csv')

                df = read_table_data_pandas(file_path)
                if df is not None:
                    analysis_result = analyze_table_data_pandas(df, args.window_size)
                    plot_table_data(analysis_result, output_image_path, args.window_size)
                    save_analysis_result(analysis_result, output_result_path)


if __name__ == "__main__":
    main()