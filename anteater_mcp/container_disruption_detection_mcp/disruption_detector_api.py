from anteater.core.ts import TimeSeries
from anteater.core.anomaly import RootCause
from typing import List
import pandas as pd
import numpy as np


def _normalize_df(df):
    cols = list(df)
    for item in cols:
        if df[item].dtype == "int64" or df[item].dtype == "float64":
            max_tmp = np.max(np.array(df[item]))
            min_tmp = np.min(np.array(df[item]))
            if max_tmp != min_tmp:
                df[item] = df[item].apply(
                    lambda x: (x - min_tmp) * 1 / (max_tmp - min_tmp)
                )


def find_discruption_source(
    self, victim_ts: TimeSeries, all_ts: List[TimeSeries]
) -> List[RootCause]:
    root_causes = []
    tmp_causes = []
    for ts in all_ts:
        # container_hostname = ts.labels.get('container_hostname', '')
        # info = self.queryContainerInfo(container_hostname) if container_hostname else {}
        # cpu_num = info.get('cpu', 0)
        cpu_num = int(ts.labels.get("cpu_num", "0"))
        if ts is victim_ts and cpu_num < 5:
            continue

        agg_data = []
        for i in range(len(victim_ts.time_stamps)):
            data = {"victim": victim_ts.values[i], "source": ts.values[i]}
            agg_data.append(data)

        agg_data_df = pd.DataFrame(agg_data)
        _normalize_df(agg_data_df)

        # metrics_correlation = agg_data_df.corr(method="spearman")
        # metrics_correlation = agg_data_df.corr(method="kendall")
        metrics_correlation = agg_data_df.corr(method="pearson")

        sorted_metrics_correlation = abs(metrics_correlation.iloc[0]).sort_values(
            ascending=False
        )
        # print("sorted_metrics_correlation:", sorted_metrics_correlation)

        causes = {
            "score": round(sorted_metrics_correlation.values[-1], 3),
            "cpu_num": cpu_num,
            "metric": ts.metric,
            "labels": ts.labels,
        }
        causes["labels"]["cpu_num"] = cpu_num

        if causes["score"] > 0.5:
            tmp_causes.append(causes)

        # root_causes.append(RootCause(
        #     metric=ts.metric,
        #     labels=ts.labels,
        #     score=round(sorted_metrics_correlation.values[-1], 3)))

    tmp_causes.sort(key=lambda x: (x["labels"]["cpu_num"], x["score"]), reverse=True)

    root_causes = [
        RootCause(
            metric=causes["metric"], labels=causes["labels"], score=causes["score"]
        )
        for causes in tmp_causes
    ]

    # print("root_causes:", root_causes)

    return root_causes[:3]
