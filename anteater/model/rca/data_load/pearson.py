import collections

import numpy as np
from scipy.stats import pearsonr


def pearson_relevant_degree(df, standard_series):
    relevant_degree = []
    for col in df.columns:
        if col == "timestamp":
            continue
        correlation_coef = pearsonr(standard_series, df[col])[0]
        relevant_degree.append((col, correlation_coef))
    relevant_degree.sort(key=lambda x: -abs(x[1]))
    return relevant_degree


def pearson_correlation(metric, df, standard_series, top_n=1):
    """Calculates the Pearson correlation coefficient with a given standard series"""
    kpi_record = collections.defaultdict(list)
    relevant_degree = pearson_relevant_degree(df, standard_series)
    for idx, item in enumerate(relevant_degree, 1):
        if idx <= top_n:
            kpi_record[metric].append(item)

    return kpi_record


def select_relevant_kpi(df, standard_series, topk=120):
    """Selects the topk metrics with the greatest Pearson correlation coefficient for a given standard series"""
    kpi_record = {}
    relevant_degree = pearson_relevant_degree(df, standard_series)

    for idx, item in enumerate(relevant_degree, 1):
        if idx <= topk:
            tmp = 0.0 if np.isnan(item[1]) else item[1]
            kpi_record[item[0]] = tmp

    return kpi_record
