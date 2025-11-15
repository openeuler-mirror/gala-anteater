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
from typing import Dict, List, Tuple
from abc import ABC
from datetime import datetime
import pytz
from math import floor
from itertools import groupby

import requests
import numpy as np
import pandas as pd

from anteater.core.ts import TimeSeries
from anteater.core.anomaly import Anomaly, RootCause
from anteater.core.kpi import KPI, ModelConfig, Feature
from anteater.utils.common import divide, GlobalVariable
from anteater.utils.datetime import DateTimeManager as dt
from anteater.utils.timer import timer
from anteater.utils.log import logger
from anteater.source.metric_loader import MetricLoader
from anteater.model.detector.base import Detector
from anteater.model.algorithms.n_sigma import n_sigma_ex
from anteater.model.algorithms.normalization import Normalization
from anteater.model.algorithms.spot import Spot
from anteater.model.algorithms.ts_dbscan import TSDBSCAN


class ContainerDisruptionDetector(Detector):
    def __init__(self, data_loader: MetricLoader, config: ModelConfig, **kwargs):
        """The detector base class initializer"""
        logger.info("========== 初始化ContainerDisruptionDetector ==========")
        super().__init__(data_loader, **kwargs)
        self.config = config
        self.q = 1e-3
        self.level = 0.98
        self.smooth_win = 3

        self.container_num = 0
        self.start_time = None
        self.end_time = None

        logger.info(
            f"配置参数: q={self.q}, level={self.level}, smooth_win={self.smooth_win}"
        )
        logger.info(f"ModelConfig params: {config.params if config else 'None'}")
        logger.info("========== ContainerDisruptionDetector初始化完成 ==========")

    @timer
    def _execute(
        self, kpis: List[KPI], features: List[Feature], **kwargs
    ) -> List[Anomaly]:
        # logger.info(f'容器干扰检测器使用的kpis{kpis}')
        logger.info("Execute cdt model: %s.", self.__class__.__name__)
        anomalies = self.detect_and_rca(kpis)

        logger.info(f"【模块1-出口】检测完成，返回异常数量: {len(anomalies)}")
        logger.info("=" * 80)
        return anomalies

    def detect_and_rca(self, kpis: List[KPI]):
        logger.info("\n" + "=" * 80)
        logger.info("【模块2-检测与根因分析】进入detect_and_rca()")

        # 获取时间范围
        start, end = dt.last(minutes=20)
        logger.info(f"时间范围: start={start}, end={end} (最近20分钟)")

        # 打印KPI详细信息
        logger.info(f"待检测KPI总数: {len(kpis)}")
        for idx, k in enumerate(kpis):
            logger.info(
                f"  KPI[{idx}]: metric={k.metric}, entity_name={k.entity_name}, params={k.params}"
            )

        # 获取机器ID列表
        logger.info("【模块2.1-获取机器列表】开始获取唯一机器ID...")
        machine_ids = self.get_unique_machine_id(start, end, kpis)
        logger.info(f"检测start: {start}, end: {end}")
        anomalies = []
        for machine_idx, _id in enumerate(machine_ids):
            logger.info(
                f"\n--- 处理机器 [{machine_idx + 1}/{len(machine_ids)}]: {_id} ---"
            )

            for kpi_idx, kpi in enumerate(kpis):
                logger.info(f"  检测KPI [{kpi_idx + 1}/{len(kpis)}]: {kpi.metric}")
                detected_anomalies = self.detect_signal_kpi(kpi, _id)
                logger.info(f"  检测到 {len(detected_anomalies)} 个异常")
                anomalies.extend(detected_anomalies)

        logger.info(
            f"\n【模块2-统计信息】总机器数: {len(machine_ids)}, 总容器数: {self.container_num}"
        )
        logger.info(f"【模块2-统计信息】总异常数: {len(anomalies)}")
        self.container_num = 0

        logger.info("【模块2-检测与根因分析】detect_and_rca()执行完成")
        logger.info("=" * 80 + "\n")
        return anomalies

    def detect_signal_kpi(self, kpi, machine_id: str) -> List[Anomaly]:
        """Detects kpi based on signal time series anomaly detection model"""
        logger.info(f"【模块3-单KPI检测】进入detect_signal_kpi()")
        logger.info(f"  参数: machine_id={machine_id}, kpi.metric={kpi.metric}")

        anomalies = []
        logger.info("【模块3.1-SPOT检测】调用detect_by_spot()进行SPOT算法检测...")
        anomalies_spot = self.detect_by_spot(kpi, machine_id)
        logger.info(
            f"【模块3.1-SPOT检测】SPOT检测完成，发现 {len(anomalies_spot)} 个异常"
        )

        if anomalies_spot:
            anomalies.extend(anomalies_spot)
            logger.info(f"【模块3-单KPI检测】累计异常数: {len(anomalies)}")

        logger.info(
            f"【模块3-单KPI检测】detect_signal_kpi()完成，返回 {len(anomalies)} 个异常"
        )
        return anomalies

    def get_kpi_ts_list(self, metric, machine_id: str, look_back):
        logger.info(f"【模块4-数据获取】进入get_kpi_ts_list()")
        logger.info(
            f"  参数: metric={metric}, machine_id={machine_id}, look_back={look_back}分钟"
        )

        if GlobalVariable.is_test_model:
            logger.info("【模块4-测试模式】使用全局测试时间范围")
            start_time, end_time = GlobalVariable.start_time, GlobalVariable.end_time
            self.start_time = start_time
            self.end_time = end_time
            logger.info(f"  测试时间: start={start_time}, end={end_time}")

            ts_list = self.data_loader.get_metric(
                start_time, end_time, metric, machine_id=machine_id
            )
            point_count = self.data_loader.expected_point_length(start_time, end_time)

        else:
            logger.info("【模块4-生产模式】使用最近时间范围")
            start, end = dt.last(minutes=look_back)
            self.start_time = start
            self.end_time = end
            logger.info(f"  生产时间: start={start}, end={end}")

            point_count = self.data_loader.expected_point_length(start, end)
            ts_list = self.data_loader.get_metric(
                start, end, metric, machine_id=machine_id
            )

        logger.info(
            f"【模块4-数据获取】获取到 {len(ts_list)} 条时间序列，期望数据点数: {point_count}"
        )
        logger.info(f"【模块4-数据获取】get_kpi_ts_list()完成")
        return point_count, ts_list

    def detect_by_spot(self, kpi, machine_id: str) -> List[Anomaly]:
        logger.info(f"【模块5-SPOT检测】进入detect_by_spot()")
        logger.info(f"  KPI参数: {kpi.params}")

        outlier_ratio_th = kpi.params["outlier_ratio_th"]
        logger.info(f"  异常比例阈值: {outlier_ratio_th}")

        logger.info("【模块5.1-计算SPOT分数】调用cal_spot_score()...")
        ts_scores = self.cal_spot_score(kpi.metric, machine_id, **kpi.params)

        if not ts_scores:
            logger.warning(
                f"【模块5-SPOT检测】关键指标 {kpi.metric} 在机器 {machine_id} 上为空!"
            )
            return []

        logger.info(f"【模块5.2-过滤结果】原始结果数: {len(ts_scores)}")
        ts_scores = [t for t in ts_scores if t[1] >= outlier_ratio_th]
        logger.info(
            f"【模块5.2-过滤结果】过滤后结果数(score >= {outlier_ratio_th}): {len(ts_scores)}"
        )

        logger.info("【模块5.3-构建异常对象】将检测结果转换为Anomaly对象...")
        anomalies = [
            Anomaly(
                machine_id=machine_id,
                metric=_ts.metric,
                labels=_ts.labels,
                score=float(_score),
                entity_name=kpi.entity_name,
                root_causes=_root_causes,
                details={"event_source": "spot", "info": _extra_info},
            )
            for _ts, _score, _extra_info, _root_causes in ts_scores
        ]

        logger.info(
            f"【模块5-SPOT检测】detect_by_spot()完成，返回 {len(anomalies)} 个异常"
        )
        return anomalies

    def cal_spot_score(
        self, metric, machine_id: str, **kwargs
    ) -> List[Tuple[TimeSeries, int, Dict, List[RootCause]]]:
        """Calculates metrics' ab score based on n-sigma method"""
        look_back = kwargs.get("look_back")  # 回溯时间窗口
        obs_size = kwargs.get("obs_size")  # 观察窗口大小
        ts_dbscan_detector = TSDBSCAN(kwargs)

        logger.info("【模块6.2-获取时间序列】调用get_kpi_ts_list()获取数据...")
        point_count, ts_list = self.get_kpi_ts_list(metric, machine_id, look_back)
        logger.info(f"【模块6.2-获取时间序列】成功获取 {len(ts_list)} 条时间序列")
        logger.info(f"  期望数据点数: {point_count}")

        ts_scores = []
        root_causes = []
        extra_info = {}

        logger.info(
            f"【模块6.3-容器统计】机器 {machine_id} 上检测到 {len(ts_list)} 个容器"
        )
        self.container_num += len(ts_list)
        logger.info(f"容器干扰检测 ts_list: {ts_list}")
        # for i, ts in enumerate(ts_list):
        #     logger.info(f"TimeSeries {i}:")
        #     logger.info(f"  labels: {ts.labels}")
        #     logger.info(f"  time_stamps: {ts.time_stamps}")
        #     logger.info(f"  values: {ts.values}")
        for _ts in ts_list:
            # import pdb;pdb.set_trace()
            detect_result = ts_dbscan_detector.detect(_ts.values)
            logger.info(f"  DBSCAN检测结果长度: {len(detect_result)}")
            logger.info(
                f"  正常点数: {sum(1 for x in detect_result if x == 0)}, "
                f"异常点数: {sum(1 for x in detect_result if x != 0)}"
            )

            if len(detect_result) != len(_ts.values):
                logger.error(
                    f"  【错误】检测结果长度不匹配! detect={len(detect_result)}, ts={len(_ts.values)}"
                )
                raise ValueError("Detect result length mismatch")

            # 根据9:1比例计算测试数据大小
            total_points = len(_ts.values)
            new_obs_size = max(1, int(total_points * 0.1))  # 至少保留1个点作为测试数据
            if obs_size is not None:
                # 如果配置中指定了obs_size，则使用配置值和计算值中的较大值
                new_obs_size = max(obs_size, new_obs_size)

            # 分离训练和测试数据
            logger.info("  【步骤6.4.2-数据分离】分离训练数据和测试数据...")
            train_data = [
                _ts.values[i]
                for i in range(len(detect_result))
                if detect_result[i] == 0
            ]
            test_data = _ts.values[-new_obs_size:]
            logger.info(
                f"  训练数据长度: {len(train_data)}, 测试数据长度: {len(test_data)} (按9:1划分)"
            )

            # 去重统计
            logger.info("  【步骤6.4.3-数据去重检查】检查数据重复度...")
            dedup_values = [k for k, g in groupby(test_data)]
            train_dedup_values = [k for k, g in groupby(train_data)]
            logger.info(
                f"  去重后训练数据: {len(train_dedup_values)}, 去重后测试数据: {len(dedup_values)}"
            )
            logger.info(
                f"  训练数据重复率: {(1 - len(train_dedup_values) / len(train_data)) * 100:.2f}% "
                f"测试数据重复率: {(1 - len(dedup_values) / len(test_data)) * 100:.2f}%"
            )

            # 过滤条件检查
            logger.info("  【步骤6.4.4-数据质量检查】检查时间序列是否满足分析条件...")
            skip_reasons = []
            if sum(_ts.values) == 0:
                skip_reasons.append("所有值为0")
            if np.max(_ts.values) < 1e3:
                skip_reasons.append(f"最大值过小({np.max(_ts.values):.2f} < 1000)")
            if len(_ts.values) < point_count * 0.6:
                skip_reasons.append(
                    f"数据点过少({len(_ts.values)} < {point_count * 0.6})"
                )
            if len(_ts.values) > point_count * 1.5:
                skip_reasons.append(
                    f"数据点过多({len(_ts.values)} > {point_count * 1.5})"
                )
            if all(x == _ts.values[0] for x in _ts.values):
                skip_reasons.append("所有值相同")
            # if len(dedup_values) < obs_size * 0.8:
            #     skip_reasons.append(f"测试数据重复度过高({len(dedup_values)} < {obs_size * 0.8})")

            if skip_reasons:
                logger.info(f"  【跳过】时间序列不符合条件: {', '.join(skip_reasons)}")
                score = 0
            else:
                logger.info(
                    "  【步骤6.4.5-SPOT算法执行】时间序列通过质量检查，开始SPOT检测..."
                )

                # 准备数据
                logger.info("    步骤A: 转换为pandas.Series...")
                ts_series = pd.Series(_ts.values)
                ts_series_train = pd.Series(train_data)

                logger.info("    步骤B: 检查边界类型...")
                ts_series_list = self._check_bound_type("upper_bound", ts_series)
                ts_series_train_list = self._check_bound_type(
                    "upper_bound", ts_series_train
                )
                logger.info(f"    边界类型数量: {len(ts_series_list)}")

                logger.info(f"    步骤C: 初始化结果数组，大小={obs_size}")
                result = np.zeros((obs_size,), dtype=np.int32)

                for bound_idx, (_ts_series, _ts_series_train) in enumerate(
                    zip(ts_series_list, ts_series_train_list)
                ):
                    logger.info(
                        f"\n    >> 处理边界 [{bound_idx + 1}/{len(ts_series_list)}] <<"
                    )
                    _ts_series_test = _ts_series[-obs_size:]

                    # 数据预处理
                    logger.info(f"    步骤D: 数据预处理...")
                    _ts_series_train = _ts_series_train.values
                    logger.info(f"      训练数据形状: {_ts_series_train.shape}")

                    # 添加噪声
                    logger.info(f"    步骤E: 添加微小噪声避免数值问题...")
                    noise_data = np.random.normal(
                        0, scale=1e-6, size=_ts_series_train.shape
                    )
                    _ts_series_train += noise_data

                    # 检查峰值数据
                    if self._is_peak_empty(_ts_series_train):
                        logger.warning(f"    【警告】峰值数据全部相同，添加更强噪声...")
                        if np.max(_ts_series_train) != 0:
                            noise_ratio = (
                                np.random.randint(
                                    -1e5, 1e5, size=_ts_series_train.shape
                                )
                                / 1e6
                            )
                            noise_data = noise_ratio * _ts_series_train
                            _ts_series_train += noise_data
                            logger.info(f"      已添加相对噪声")

                    # 归一化
                    logger.info(f"    步骤F: 归一化训练数据...")
                    _ts_series_train, mean, std = Normalization.clip_transform(
                        _ts_series_train[np.newaxis, :], is_clip=False
                    )
                    _ts_series_train = _ts_series_train[0]
                    logger.info(f"      归一化参数: mean={mean}, std={std}")

                    # 初始化SPOT模型
                    logger.info(f"    步骤G: 初始化SPOT模型(q={self.q})...")
                    spot = Spot(q=self.q)
                    level = self._check_level(_ts_series_train, self.level)
                    logger.info(f"      调整后的level: {level}")
                    spot.initialize(_ts_series_train, level=level)

                    # 预测
                    logger.info(f"    步骤H: 对测试数据进行预测...")
                    _ts_series_test = _ts_series_test.values
                    _ts_series_test, _, _ = Normalization.clip_transform(
                        _ts_series_test[np.newaxis, :],
                        mean=mean,
                        std=std,
                        is_clip=False,
                    )
                    _ts_series_test = _ts_series_test[0]

                    logger.info(f"    步骤I: 运行SPOT检测...")
                    thr_with_alarms = spot.run(_ts_series_test, with_alarm=True)
                    bound_result = np.array(
                        _ts_series_test > thr_with_alarms["thresholds"], dtype=np.int32
                    )
                    logger.info(f"      本次边界检测到异常点数: {np.sum(bound_result)}")
                    result += bound_result

                output = np.sum(result)
                logger.info(
                    f"  【步骤6.4.6-结果统计】SPOT总检测结果: {result.tolist()}"
                )
                logger.info(f"  累计异常点数: {output}/{obs_size}")

                # 判断是否异常
                if output >= 3:
                    logger.info(f"  【检测到异常】异常点数({output}) >= 阈值(3)")
                    logger.info(f"    原始数据: {_ts.values}")
                    logger.info(f"    SPOT结果: {result}")

                    container_hostname = _ts.labels.get("container_name", "")
                    machine_id = _ts.labels.get("machine_id", "")
                    logger.info(
                        f"    容器信息: container_name={container_hostname}, machine_id={machine_id}"
                    )

                    # ---------- 记录异常时间段 ----------
                    detect_mask = result > 0
                    test_timestamps = _ts.time_stamps[-obs_size:]

                    abnormal_points = []
                    for i in range(obs_size):
                        if detect_mask[i]:
                            ts = test_timestamps[i]
                            # 强制转 datetime
                            if isinstance(ts, str):
                                try:
                                    ts = datetime.fromisoformat(
                                        ts.replace("Z", "+00:00")
                                    )
                                except:
                                    ts = None
                            abnormal_points.append(ts)

                    # 初始化结构
                    extra_info = {}

                    if abnormal_points:
                        extra_info["abnormal_start"] = min(
                            [t for t in abnormal_points if t]
                        )
                        extra_info["abnormal_end"] = max(
                            [t for t in abnormal_points if t]
                        )
                    else:
                        extra_info["abnormal_start"] = None
                        extra_info["abnormal_end"] = None

                    # 合并 container_extra_data
                    container_extra = self.get_container_extra_info(
                        machine_id,
                        container_hostname,
                        self.start_time,
                        self.end_time,
                        obs_size,
                    )

                    extra_info.update(container_extra)

                    logger.info(f"    容器额外信息: {extra_info}")

                    logger.info(
                        "  【步骤6.4.8-根因分析】调用find_disruption_source()..."
                    )
                    root_causes = self.find_disruption_source(_ts, ts_list)
                    logger.info(f"    根因数量: {len(root_causes)}")
                    for rc_idx, rc in enumerate(root_causes):
                        logger.info(
                            f"      根因[{rc_idx + 1}]: metric={rc.metric}, score={rc.score}, "
                            f"labels={rc.labels}"
                        )
                else:
                    logger.info(f"  【正常】异常点数({output}) < 阈值(3)，判定为正常")

                score = divide(output, obs_size)
                logger.info(f"  最终得分: {score:.4f}")

            ts_scores.append((_ts, score, extra_info, root_causes))

        logger.info(f"【模块6-SPOT分数计算】cal_spot_score()完成")
        logger.info(f"  总计处理: {len(ts_scores)} 条时间序列")
        logger.info(
            f"  异常数量: {sum(1 for _, score, _, _ in ts_scores if score > 0)}"
        )
        logger.info("-" * 80 + "\n")
        return ts_scores

    def find_disruption_source(
        self, victim_ts: TimeSeries, all_ts: List[TimeSeries]
    ) -> List[RootCause]:
        logger.info(f"【模块7-根因分析】进入find_disruption_source()")
        logger.info(f"  受害者时间序列: {victim_ts.labels}")
        logger.info(f"  候选源时间序列数量: {len(all_ts)}")

        root_causes = []
        tmp_causes = []

        logger.info("【模块7.1-遍历候选源】分析每个候选源与受害者的相关性...")
        for ts_idx, ts in enumerate(all_ts):
            cpu_num = int(ts.labels.get("cpu_num", "0"))

            # 跳过受害者自身（如果CPU数小于5）
            if ts is victim_ts and cpu_num < 5:
                logger.info(f"  候选源[{ts_idx}]: 跳过（受害者自身且CPU数<5）")
                continue

            logger.info(f"  候选源[{ts_idx}]: {ts.labels}, cpu_num={cpu_num}")

            # 构建数据对
            logger.info(f"    【步骤7.1.1-数据对齐】构建受害者与候选源的数据对...")
            agg_data = []
            for i in range(len(victim_ts.time_stamps)):
                data = {"victim": victim_ts.values[i], "source": ts.values[i]}
                agg_data.append(data)

            agg_data_df = pd.DataFrame(agg_data)
            logger.info(f"    数据对数量: {len(agg_data_df)}")

            # 归一化
            logger.info(f"    【步骤7.1.2-数据归一化】对数据进行归一化...")
            self._normalize_df(agg_data_df)

            # 计算相关性
            logger.info(f"    【步骤7.1.3-相关性计算】计算Pearson相关系数...")
            metrics_correlation = agg_data_df.corr(method="pearson")
            sorted_metrics_correlation = abs(metrics_correlation.iloc[0]).sort_values(
                ascending=False
            )
            correlation_score = round(sorted_metrics_correlation.values[-1], 3)
            logger.info(f"    相关性得分: {correlation_score}")

            causes = {
                "score": correlation_score,
                "cpu_num": cpu_num,
                "metric": ts.metric,
                "labels": ts.labels,
            }
            causes["labels"]["cpu_num"] = cpu_num

            if causes["score"] > 0.5:
                logger.info(f"    【符合条件】相关性 > 0.5，加入候选根因列表")
                tmp_causes.append(causes)
            else:
                logger.info(f"    【不符合条件】相关性 <= 0.5，跳过")

        logger.info(f"【模块7.2-排序筛选】对候选根因按CPU数和得分排序...")
        logger.info(f"  排序前候选数: {len(tmp_causes)}")
        tmp_causes.sort(
            key=lambda x: (x["labels"]["cpu_num"], x["score"]), reverse=True
        )

        logger.info("【模块7.3-构建根因对象】转换为RootCause对象...")
        root_causes = [
            RootCause(
                metric=causes["metric"], labels=causes["labels"], score=causes["score"]
            )
            for causes in tmp_causes
        ]

        logger.info(f"【模块7-根因分析】find_disruption_source()完成")
        logger.info(f"  最终返回前3个根因，总候选数: {len(root_causes)}")
        if root_causes[:3]:
            for rc_idx, rc in enumerate(root_causes[:3]):
                logger.info(
                    f"    根因[{rc_idx + 1}]: {rc.metric}, score={rc.score}, cpu_num={rc.labels.get('cpu_num')}"
                )

        return root_causes[:3]

    def get_container_extra_info(
        self,
        machine_id: str,
        container_name: str,
        start_time: datetime,
        end_time: datetime,
        obs_size: int,
    ) -> Dict:
        logger.info(f"【模块8-容器额外信息】进入get_container_extra_info()")
        logger.info(f"  参数: machine_id={machine_id}, container_name={container_name}")
        logger.info(f"  时间范围: {start_time} ~ {end_time}, obs_size={obs_size}")

        extra_metrics = self.config.params.get("extra_metrics", "").split(",")
        logger.info(f"  需要获取的额外指标: {extra_metrics}")

        result = {"container_name": container_name, "machine_id": machine_id}

        logger.info("【模块8.1-获取额外指标】遍历每个额外指标...")
        for metric_idx, metric in enumerate(extra_metrics):
            logger.info(f"  处理指标[{metric_idx + 1}/{len(extra_metrics)}]: {metric}")

            ts_list = self.data_loader.get_metric(
                start_time, end_time, metric, machine_id=machine_id
            )
            logger.info(f"    获取到 {len(ts_list)} 条时间序列")

            for _ts in ts_list:
                if container_name == _ts.labels.get("container_name", ""):
                    logger.info(f"    【匹配】找到对应容器的时间序列")
                    values = _ts.values
                    logger.info(f"      数据长度: {len(values)}")

                    logger.info(f"    【步骤8.1.1-趋势计算】计算指标趋势...")
                    trend = self.cal_trend(values, obs_size)
                    logger.info(f"      趋势值: {trend}")

                    result[metric] = trend
                    result["appkey"] = _ts.labels.get("appkey", "")
                    result["cpu_num"] = int(_ts.labels.get("cpu_num", "0"))
                    logger.info(
                        f"      appkey={result['appkey']}, cpu_num={result['cpu_num']}"
                    )
                    break

        logger.info(f"【模块8-容器额外信息】get_container_extra_info()完成")
        logger.info(f"  返回结果: {result}")
        return result

    @staticmethod
    def _normalize_df(df):
        logger.info("    【辅助函数-归一化】_normalize_df()执行")
        cols = list(df)
        logger.info(f"      列数: {len(cols)}")

        for item in cols:
            if df[item].dtype == "int64" or df[item].dtype == "float64":
                max_tmp = np.max(np.array(df[item]))
                min_tmp = np.min(np.array(df[item]))
                logger.info(f"      列'{item}': max={max_tmp}, min={min_tmp}")

                if max_tmp != min_tmp:
                    df[item] = df[item].apply(
                        lambda x: (x - min_tmp) * 1 / (max_tmp - min_tmp)
                    )
                    logger.info(f"        已归一化")
                else:
                    logger.info(f"        跳过（max == min）")

    @staticmethod
    def _check_bound_type(bound_type, metric_data):
        logger.info(
            f"    【辅助函数-边界类型】_check_bound_type(), bound_type={bound_type}"
        )

        if bound_type == "bi_bound":
            data = metric_data, -metric_data
            logger.info(f"      返回双边界")
        elif bound_type == "lower_bound":
            data = (-metric_data,)
            logger.info(f"      返回下边界")
        else:
            data = (metric_data,)
            logger.info(f"      返回上边界")

        return data

    @staticmethod
    def _check_level(metric_data, level):
        logger.info(f"    【辅助函数-Level检查】_check_level(), level={level}")
        data_size = len(metric_data)

        if int(data_size * (1 - level)) == 0:
            peak = 2
            level = 1.0 - peak / float(data_size) - 1e-6
            logger.info(f"      调整level: {level} (数据量={data_size})")

        return level

    def _is_peak_empty(self, metric_data):
        logger.info(f"    【辅助函数-峰值检查】_is_peak_empty()")
        data_size = len(metric_data)
        sort_data = np.sort(metric_data)
        level = self.level - floor(self.level)
        peak_num = int(level * data_size)

        if peak_num == 0:
            peak_num = min(2, data_size)

        init_threshold = sort_data[peak_num]
        peaks = metric_data[metric_data > init_threshold]

        is_empty = peaks.size == 0
        logger.info(f"      峰值数量: {peaks.size}, is_empty={is_empty}")

        return is_empty

    def _check_smooth(self, metric_data):
        logger.info(f"    【辅助函数-平滑窗口】_check_smooth()")
        data_size = len(metric_data)

        if data_size < self.smooth_win:
            smooth_win = data_size // 2
            logger.info(f"      数据量小，调整窗口: {smooth_win}")
        else:
            smooth_win = self.smooth_win
            logger.info(f"      使用默认窗口: {smooth_win}")

        return smooth_win

    @staticmethod
    def cal_trend(metric_values: list, obs_size: int) -> float:
        logger.info(f"    【辅助函数-趋势计算】cal_trend()")
        logger.info(f"      数据长度: {len(metric_values)}, obs_size={obs_size}")

        pre = metric_values[:-obs_size]
        check = metric_values[-obs_size:]

        pre_mean = np.mean(pre)
        check_mean = np.mean(check)

        logger.info(f"      历史均值: {pre_mean:.2f}, 观测期均值: {check_mean:.2f}")

        trend = (check_mean - pre_mean) / pre_mean if pre_mean > 0 else 0.0
        result = round(trend, 3)

        logger.info(
            f"      趋势值: {result} ({'+' if result > 0 else ''}{result * 100:.1f}%)"
        )

        return result
