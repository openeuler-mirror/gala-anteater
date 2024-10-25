# coding=utf-8

from sklearn.cluster import DBSCAN
from anteater.model.algorithms.slow_node_algs.space_comp_detector.sliding_window_dbscan import SlidingWindowDBSCAN
from anteater.model.algorithms.slow_node_algs.space_comp_detector.outlier_data_detector import OuterDataDetector

space_node_detectors = {
    "OuterDataDetector": OuterDataDetector,
    "SlidingWindowDBSCAN": SlidingWindowDBSCAN,
}
