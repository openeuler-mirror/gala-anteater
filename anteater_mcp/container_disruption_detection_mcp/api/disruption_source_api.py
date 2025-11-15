from __future__ import annotations
from typing import List
import pandas as pd
import numpy as np
import logging

from anteater.core.ts import TimeSeries
from anteater.core.anomaly import RootCause

logger = logging.getLogger("disruption_source_api")


class DisruptionSourceAPI:
    """
    容器干扰源分析 API
    输入：
        - victim_ts：被干扰容器对应的时间序列
        - all_ts：同机所有容器的同指标时间序列
    输出：
        - RootCause 列表（按相关性排序）
    """

    # 数据归一化（0-1）
    @staticmethod
    def _normalize_df(df: pd.DataFrame):
        """对列进行 min-max 归一化（安全检查版）"""
        for col in df.columns:
            if df[col].dtype in ("int64", "float64", "float32", "int32"):
                col_min = df[col].min()
                col_max = df[col].max()
                if col_max != col_min:
                    df[col] = (df[col] - col_min) / (col_max - col_min)
                else:
                    # 无波动，归一化也无意义，保持原样即可
                    continue

    # 核心：查找干扰源
    def find_sources(
        self,
        victim_ts: TimeSeries,
        all_ts: List[TimeSeries],
        *,
        min_corr: float = 0.5,
        topk: int = 3,
    ) -> List[RootCause]:
        """
        返回按相关性排序的 RootCause 列表。
        """

        logger.info(f"[RCA] 开始寻找干扰源，容器={victim_ts.labels}")

        if not victim_ts or not all_ts:
            logger.warning("[RCA] 输入数据为空，无法分析")
            return []

        if len(victim_ts.values) == 0:
            logger.warning("[RCA] victim_ts.values 为空")
            return []

        # 输出候选列表
        candidates = []

        for idx, ts in enumerate(all_ts):
            cpu_num = int(ts.labels.get("cpu_num", "0"))

            # 原始规则：受害者自身且 cpu<5，跳过
            if ts is victim_ts and cpu_num < 5:
                logger.debug(f"[RCA] 跳过自身（CPU < 5），labels={ts.labels}")
                continue

            # 长度安全检查（必须等长）
            if len(ts.values) != len(victim_ts.values):
                logger.warning(
                    f"[RCA] ts 与 victim_ts 长度不一致，skip，ts_len={len(ts.values)}, victim_len={len(victim_ts.values)}"
                )
                continue

            # 构建 DataFrame
            try:
                df = pd.DataFrame(
                    {
                        "victim": victim_ts.values,
                        "source": ts.values,
                    }
                )
            except Exception as e:
                logger.warning(f"[RCA] 构建 DataFrame 失败，skip，err={e}")
                continue

            # 归一化
            self._normalize_df(df)

            # Pearson 相关性
            try:
                corr = df.corr(method="pearson")
                sorted_corr = abs(corr.iloc[0]).sort_values(ascending=False)
                score = float(round(sorted_corr.values[-1], 3))
            except Exception as e:
                logger.warning(f"[RCA] 计算相关性失败：{e}")
                continue

            entry = dict(
                metric=ts.metric,
                labels=dict(ts.labels),
                score=score,
                cpu_num=cpu_num,
            )
            entry["labels"]["cpu_num"] = cpu_num

            # 过滤低相关性
            if score >= min_corr:
                candidates.append(entry)
                logger.info(
                    f"[RCA] 候选源加入：metric={ts.metric}, score={score}, cpu={cpu_num}"
                )

        # 排序：CPU数优先，其次相关性
        candidates.sort(
            key=lambda x: (x["labels"]["cpu_num"], x["score"]),
            reverse=True,
        )

        logger.info(f"[RCA] 相关候选源总数: {len(candidates)}")

        # 转换为 RootCause 列表
        root_causes = [
            RootCause(metric=c["metric"], labels=c["labels"], score=c["score"])
            for c in candidates[:topk]
        ]

        return root_causes
