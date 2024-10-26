# coding=utf-8

from anteater.model.algorithms.slow_node_algs.time_comp_detector.sliding_window_n_sigma_detector import \
    SlidingWindowKSigmaDetector
from anteater.model.algorithms.slow_node_algs.time_comp_detector.ts_dbscan_detector import TSDBSCANDetector

time_node_detectors = {
    "TSDBSCANDetector": TSDBSCANDetector,
    "SlidingWindowKSigmaDetector": SlidingWindowKSigmaDetector
}
