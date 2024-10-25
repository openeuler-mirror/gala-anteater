import math
from collections import Counter

import numpy as np
from sklearn.cluster import DBSCAN
from anteater.utils.log import logger


class SlidingWindowDBSCAN():
    def __init__(self, cfg):
        self.smooth_window = cfg.get("smooth_window")
        self.smooth = cfg.get("smooth")
        self.cv_threshold = cfg.get("cv_threshold")  # 离散系数
        self.eps = cfg.get("eps")
        self.min_samples = cfg.get("min_samples")
        self.window_size = cfg.get("window_size")
        self.scaling = cfg.get("scaling")
        # 距离函数为cosine或者euclidean, 从以下字符串选：'sokalsneath', 'correlation', 'sokalmichener',
        # 'kulsinski', 'jaccard', 'wminkowski', 'hamming', 'l1', 'euclidean', 'chebyshev', 'seuclidean',
        # 'mahalanobis', 'braycurtis', 'minkowski', 'haversine', 'cityblock', 'matching',
        # 'yule', 'canberra', 'dice', 'rogerstanimoto', 'russellrao', 'l2',
        # 'precomputed', 'nan_euclidean', 'sqeuclidean', 'manhattan', 'cosine'

        self.dist_metric = cfg.get("dist_metric", "euclidean")
        self.buffer = None
        self.buffer_size = None
        self.cursor = None

    @staticmethod
    def _scaling_normalization(clustering_data: np.ndarray):
        max_val = np.max(clustering_data)
        if max_val > 100 or max_val < 1:
            scale_factor = 10 ** math.ceil(math.log10(max_val / 100))
        else:
            scale_factor = 1
        normalized_values = clustering_data / scale_factor
        return normalized_values

    def _update_buffer(self, data: np.ndarray) -> None:
        if self.buffer is None:
            return

        self.buffer[:, self.cursor] = data
        self.buffer_size = self.buffer_size + 1
        self.cursor = (self.cursor + 1) % self.window_size

        if self.buffer_size >= self.window_size:
            self.buffer_size = self.window_size

    def _db_scan(self, data) -> np.ndarray:
        data = np.swapaxes(data, 0, 1)
        # 对data取均值
        compute_data = np.mean(data, axis=-1)
        if self.scaling:
            # 这里是针对processMEM专门调的规则，其他指标空间对比设置scaling为false，不走这里的逻辑，保持原样，从而限制影响只有processMEM
            compute_data = np.mean(data, axis=-1)
            rate = abs(np.max(compute_data) - np.min(compute_data)) / max(2 * abs(np.median(compute_data)), 1e-8)
            if rate > 0.2:
                compute_data = compute_data / max(np.mean(compute_data), 1e-8)
            else:
                compute_data = compute_data / max(2 * np.max(compute_data), 1e-8)
        if len(compute_data.shape) == 1:
            compute_data = np.expand_dims(compute_data, axis=-1)
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.dist_metric).fit_predict(compute_data)
        label_counts = Counter(labels)
        # 找到样本数量最多的类别
        most_common_label, _ = label_counts.most_common(1)[0]
        new_labels = np.where(labels == most_common_label, 0, 1)
        broad_cast_labels = np.broadcast_to(new_labels, (data.shape[1], new_labels.size))
        return broad_cast_labels

    @staticmethod
    def _coefficient_of_variation(threshold, data):
        # 一组数内最大值与最小值之差的一半除法一组数的中间值。
        rate = abs(np.max(data) - np.min(data)) / max(2 * abs(np.median(data)), 1e-8)
        cv_label = rate > threshold
        logger.debug("coefficient_of_variation:%s, threshold:%s", rate, threshold)
        cv_label_broadcast = np.ones_like(data) * cv_label
        return cv_label_broadcast

    def detect(self, test_data: np.ndarray):
        obj_num = test_data.shape[1]
        # 第一维为对象维度，第二维为对象指标维度
        self.buffer = np.zeros((obj_num, self.window_size))
        self.buffer_size = 0
        self.cursor = 0
        if self.smooth:
            test_data = np.apply_along_axis(
                lambda m: np.convolve(m, np.ones(self.window_size) / self.window_size, mode='same'), axis=0,
                arr=test_data)
        ret_values = np.zeros(test_data.shape)
        for i in range(test_data.shape[0], 0, -self.window_size):
            start_index = max(0, i - self.window_size)
            detect_data = test_data[start_index:start_index + self.window_size]
            if self.scaling:
                test_data = self._scaling_normalization(test_data)
            if len(detect_data) < self.window_size:
                continue
            label_de_scan = self._db_scan(detect_data)
            label_cv = self._coefficient_of_variation(self.cv_threshold, detect_data)
            label = np.logical_and(label_de_scan, label_cv)
            ret_values[start_index:start_index + self.window_size, :] = label
            if np.any(label):
                logger.debug(detect_data)
                logger.debug(label)
        return ret_values
