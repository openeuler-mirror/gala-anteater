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

import os
import random
from multiprocessing import Process, Event, Queue
from threading import Thread
from typing import Callable

import cvxpy
import numpy as np
import scipy.fftpack as spfft
from cvxpy import SolverError
from numpy.fft import fft, ifft
from numpy.linalg import norm

from anteater.utils.common import divide
from anteater.utils.log import logger


def p_normalize(x: np.ndarray):
    """Normalization"""
    p_min = 0.05
    x_max, x_min = np.max(x), np.min(x)
    x_min *= 1 - p_min
    return divide(x - x_min, x_max - x_min)


def similarity(x, y):
    """Calculates the similarity of the two vectors"""
    return divide(1, 1 + np.sqrt(np.sum(np.square(x - y))))


def online_lesinn(incoming_data: np.ndarray, historical_data: np.ndarray,
                  t: int = 50, phi: int = 20, random_state: int = None):
    """Online outlier algorithm lesinn"""
    if random_state:
        random.seed(random_state)
        np.random.seed(random_state)
    m = incoming_data.shape[0]
    # Concatenates together all historical data and data that needs to calculate outliers
    if historical_data.shape:
        all_data = np.concatenate([historical_data, incoming_data], axis=0)
    else:
        all_data = incoming_data
    n, d = all_data.shape
    data_score = np.zeros((m,))
    sample = set()
    for i in range(m):
        score = 0
        for _ in range(t):
            sample.clear()
            while len(sample) < phi:
                sample.add(random.randint(0, n - 1))
            nn_sim = 0
            for each in sample:
                nn_sim = max(
                    nn_sim, similarity(incoming_data[i, :], all_data[each, :])
                )
            score += nn_sim
        if score:
            data_score[i] = divide(t, score)
    return data_score


def lesinn_score(incoming_data: np.ndarray, historical_data: np.ndarray,
                 random_state=None, lesinn_t=50, lesinn_phi=20):
    """Sampling confidence"""
    value = online_lesinn(incoming_data, historical_data,
                          random_state=random_state, t=lesinn_t,
                          phi=lesinn_phi)
    value = divide(1, value)
    return p_normalize(value)


def moving_average(x: np.ndarray, window: int, stride: int):
    """Sliding window average"""
    n, k = x.shape
    if window > n:
        window = n
    score = np.zeros(n)
    # Record the number of times each point is updated, and represent the weight of the original \\
    # value of the point as the impact of the new point is added
    score_weight = np.zeros(n)
    # The window starts
    wb = 0
    while True:
        # The window ends
        we = wb + window
        x_window = x[wb:we]
        # The average vector of the window
        x_mean = np.mean(x_window, axis=0)
        # Add influence to the score
        dis = np.sqrt(np.sum(np.square(x_window - x_mean), axis=1))
        score[wb:we] = (score[wb:we] * score_weight[wb:we] + dis)
        score_weight[wb:we] += 1
        score[wb:we] /= score_weight[wb:we]
        if we >= n:
            break
        wb += stride
    # Map score to (0, 1)
    return score


def online_moving_average(incoming_data: np.ndarray, historical_data: np.ndarray,
                          window: int, stride: int):
    """online sliding windows move average"""
    n = incoming_data.shape[0]
    # 根据窗口大小计算出所需要的所有数据量(把第一个窗口的终点放在incoming_data的起点)
    need_history_begin = max(0, historical_data.shape[0] - window + 1)
    need_data = np.concatenate(
        [incoming_data, historical_data[need_history_begin:]], axis=0
    )
    score = moving_average(need_data, window, stride)
    # 截取最后n个点的数据表示incoming_data的数值将score映射到(0, 1]上
    return score[0:n]


def moving_average_score(incoming_data: np.ndarray, historical_data: np.ndarray,
                         moving_average_window, moving_average_stride):
    """Moving average score"""
    value = online_moving_average(incoming_data, historical_data,
                                  moving_average_window,
                                  moving_average_stride)
    value = divide(1, 1 + value)
    return p_normalize(value)


def _ncc_c(x, y):
    den = np.array(norm(x) * norm(y))
    den[den == 0] = np.Inf

    x_len = len(x)
    fft_size = 1 << (2 * x_len - 1).bit_length()
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
    cc = np.concatenate((cc[-(x_len - 1):], cc[:x_len]))
    return np.divide(np.real(cc), den)


def _sbd(x, y):
    ncc = _ncc_c(x, y)
    idx = ncc.argmax()
    dist = 1 - ncc[idx]

    return dist


def make_leaf(label):
    """Build leaf nodes"""
    return ['leaf', label]


def make_cluster(distance, left, right):
    """ Build the central node """
    return ['cluster', distance, left, right]


def get_idx(idx):
    x = idx[0][0]
    y = idx[1][0]
    if y > x:
        temp = y
        y = x
        x = temp
    return x, y


def direct_cluster(simi_matrix):
    """Direct clustering"""
    # Uses list to build trees
    # Gets the number of points corresponding to the matrix
    n = len(simi_matrix)
    nodes = [make_leaf(label) for label in range(n)]  # Builds leaf nodes
    # Fills the diagonal with the maximum value
    np.fill_diagonal(simi_matrix, float('Inf'))
    root = 0  # Records the subscript of the root node
    while n > 1:
        # Look for the subscript of the smallest number in the similarity matrix
        idx = np.where(simi_matrix == simi_matrix.min())
        x, y = get_idx(idx)
        distance = simi_matrix[x][y]
        cluster_node = make_cluster(distance, nodes[x], nodes[y])
        nodes[y] = cluster_node
        root = y
        simi_matrix[x] = float('Inf')
        simi_matrix[:, x] = float('Inf')
        n = n - 1
    return nodes[root]


def get_type(node):
    """ Gets the type of leaf node """
    return node[0]


def get_label(node):
    """ Gets the label of the leaf node """
    if get_type(node) != 'leaf':
        raise ValueError('node is not the leaf type')
    return node[1]


def get_left(node):
    """ Gets the left subtree """
    if get_type(node) != 'cluster':
        raise ValueError('node is not the cluster type')
    return node[2]


def get_right(node):
    """ Gets the right subtree """
    if get_type(node) != 'cluster':
        raise ValueError('node is not the cluster type')
    return node[3]


def get_distance(node):
    """ Gets the distance value """
    if get_type(node) != 'cluster':
        raise ValueError('node is not the cluster type')
    return node[1]


def get_leaf_labels(node):
    """ Gets the value of the label """
    # Starting from the node node, traverse the leaf nodes
    node_type = get_type(node)
    if node_type == 'leaf':
        labels = [get_label(node)]
    elif node_type == 'cluster':
        labels = get_leaf_labels(get_left(node))
        labels.extend(get_leaf_labels(get_right(node)))
    return labels


def get_classify(distance, node):
    """ Get categorized data """
    node_type = get_type(node)
    if node_type == 'leaf':
        return [[get_label(node)]]
    elif node_type == 'cluster' and get_distance(node) < distance:
        return [get_leaf_labels(node)]
    else:
        llabels = get_classify(distance, get_left(node))
        rlabels = get_classify(distance, get_right(node))
        llabels.extend(rlabels)
        return llabels


def cluster(x: np.ndarray, threshold: float = 0.01):
    """BSD - clustering

    Parameters:
        x: numpy.ndarray shape = (m, n)
        threshold: float (default: 0.01) the threshold of direct clustering
    Return
        The return element is numpy.ndarray of numpy.ndarray, where the element is an integer \\
        number of 0 ~ (n-1), which is a class in the same list
    """
    ny, nx = x.shape
    distance = np.zeros((nx, nx))
    for (i, j), _ in np.ndenumerate(distance):
        distance[i, j] = _sbd(x[:, i], x[:, j])
    tree = direct_cluster(distance)
    return get_classify(threshold, tree)


def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T,
                     norm='ortho', axis=0)


def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T,
                      norm='ortho', axis=0)


def reconstruct(n, d, index, value):
    """Compressed sensing sampling reconstruction algorithm

    :param n: The amount of data that needs to be reconstructed
    :param d: The dimensions of the data need to be rebuilt
    :param index: The time dimension coordinates of the sample point belong to [0, n-1]
    :param value: The KPI value of the sampling point, shape=(m, d), m is the amount of sampled data
    :return:x_re: Constructed KPI data, shape=(n, d)
    """
    x = np.zeros((n, d))
    x[index, :] = value

    a = index
    b = index

    if d > 1:
        for _ in range(d - 1):
            a = a + n * 1
            b = np.concatenate((b, a))
    index = b.astype(int)

    # random sample of indices
    ri = index
    b = x.T.flat[ri]

    transform_mat = np.kron(
        spfft.idct(np.identity(d), norm='ortho', axis=0),
        spfft.idct(np.identity(n), norm='ortho', axis=0)
    )
    transform_mat = transform_mat[ri, :]  # same as phi times kron

    # do L1 optimization
    vx = cvxpy.Variable(d * n)
    objective = cvxpy.Minimize(cvxpy.norm(vx, 1))
    constraints = [transform_mat * vx == b]
    prob = cvxpy.Problem(objective, constraints)
    prob.solve(solver='OSQP')
    x_transformed = np.array(vx.value).squeeze()
    # reconstruct signal
    x_t = x_transformed.reshape(d, n).T  # stack columns
    x_re = idct2(x_t)

    # confirm solution
    if not np.allclose(b, x_re.T.flat[ri]):
        logger.warning('Values at sample indices don\'t match original.')

    return x_re


class CycleFeatureProcess(Process):
    """A process that calculates characteristics within a single cycle"""

    def __init__(self, task_queue: Queue, result_queue: Queue, cluster_threshold: float):
        """
        :param task_queue: Job queue
        :param result_queue: Result queue
        :param cluster_threshold: Cluster threshold
        """
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.cluster_threshold = cluster_threshold

    def run(self):
        logger.info('CycleFeatureProcess-%d: start', os.getpid())
        while not self.task_queue.empty():
            group_index, cycle_data = self.task_queue.get()
            self.result_queue.put(
                (group_index, cluster(cycle_data, self.cluster_threshold)))

        logger.info('CycleFeatureProcess-%d: exit', os.getpid())


class WindowReconstructProcess(Process):
    """Window rebuild worker process"""

    def __init__(self, data: np.ndarray, task_queue: Queue, result_queue: Queue,
                 cycle: int, latest_windows: int, sample_score_method: Callable,
                 sample_rate: float, scale: float, rho: float, sigma: float,
                 random_state: int, retry_limit: int, task_return_event: Event()):
        """
        :param data: A copy of the original data
        :param task_queue: Job queue
        :param result_queue: Result queue
        :param cycle: cycle
        :param latest_windows: The number of recent historical periods referenced
                               when calculating the sampled value metric
        :param sample_score_method: Method for calculating sampling value metrics
        :param sample_rate: Sample rate
        :param scale: Sampling parameters: Isometric sampling point multiplication
        :param rho: Sample parameters: Center sampling probability
        :param sigma: Sample parameters: The degree of sampling concentration
        :param random_state: Random number seed
        :param retry_limit: The upper limit of each window retry
        :param task_return_event: Events that are triggered when a job is completed
                                  and notify the main process to collect
        """
        super().__init__()
        self.data = data
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.cycle = cycle
        self.latest_windows = latest_windows
        self.sample_score_method = sample_score_method
        self.sample_rate = sample_rate
        self.scale = scale
        self.rho = rho
        self.sigma = sigma
        self.random_state = random_state
        self.retry_limit = retry_limit
        self.task_return_event = task_return_event

        self.max_seed = 10 ** 9 + 7

    @staticmethod
    def reconstruct_value(groups, n, timestamp, values, rec):
        for group in groups:
            x_re = reconstruct(
                n, len(group), timestamp,
                values[:, group]
            )
            for j, value in enumerate(group):
                rec[:, value] = x_re[:, j]
        return rec

    def run(self):
        if self.random_state:
            np.random.seed(self.random_state)
        logger.info('WindowReconstructProcess-%d: start', os.getpid())
        while not self.task_queue.empty():
            wb, we, group = self.task_queue.get()
            hb = max(0, wb - self.latest_windows)
            latest = self.data[hb:wb]
            window_data = self.data[wb:we]
            sample_score = self.sample_score_method(window_data, latest)
            rec_window, retries = \
                self.window_sample_reconstruct(
                    data=window_data,
                    groups=group,
                    random_state=self.random_state * wb * we % self.max_seed
                )
            self.result_queue.put((wb, we, rec_window, retries, sample_score))
            self.task_return_event.set()

        logger.info('WindowReconstructProcess-%d: exit', os.getpid())

    def window_sample_reconstruct(self, data: np.ndarray, groups: list, random_state: int):
        """
        :param data: raw data
        :param groups: groups
        :param random_state: Random seeds
        :return: Rebuild data, number of reconstruction attempts
        """
        n, d = data.shape
        retry_count = 0
        sample_rate = self.sample_rate
        while True:
            try:
                if random_state:
                    np.random.seed(random_state)
                    timestamp = np.random.choice(
                        np.arange(n),
                        size=int(np.round(sample_rate * n)),
                        replace=False
                    )
                    values = data[timestamp]
                else:
                    timestamp = np.random.choice(
                        np.arange(n),
                        size=int(np.round(sample_rate * n)),
                        replace=False
                    )
                    values = data[timestamp]
                rec = np.zeros(shape=(n, d))
                rec = self.reconstruct_value(groups, n, timestamp, values, rec)
                break
            except SolverError as e:
                if retry_count > self.retry_limit:
                    raise ValueError('retry failed, please try higher sample '
                                     'rate or window size') from e
                sample_rate += divide(1 - sample_rate, 4)
                retry_count += 1

                logger.warning(
                    'Reconstruct failed, retry with higher '
                    'sample rate %f, retry times remain %d',
                    sample_rate, self.retry_limit - retry_count)
        return rec, retry_count


class CSAnomalyDetector:
    """
    Offline multi-process anomaly detector based on compressed sensing sampling reconstruction
    """

    def __init__(self, cluster_threshold: float, sample_rate: float, sample_score_method,
                 distance, workers: int = 1, latest_windows: int = 96, scale: float = 5,
                 rho: float = 0.1, sigma: float = 1 / 24, random_state=None, retry_limit=10,
                 without_grouping: str = None):
        """
        :param cluster_threshold: Cluster parameter: threshold
        :param sample_rate: Sample rate
        :param sample_score_method: Sample Point Confidence Calculation Function
                                    Input (array(n * d)) Output array(n) represents
                                    the sampling confidence level of n points of the input
        :param distance: A function that calculates distance, input (array(n*d),
                         array(n*d)) output real represents the distance between two inputs
        :param workers: Count the number of threads
        :param latest_windows: The number of history windows referenced when sampling
        :param scale: Sampling parameters: Isometric sampling point multiplication
        :param rho: Sample parameters: Center sampling probability
        :param sigma: Sample parameters: The degree of sampling concentration
        :param random_state: Random number seed
        :param retry_limit: The number of retries of the solution, after which the
                            solution is still unsuccessful, an exception is thrown
        :param without_grouping: Downgrade experiments: No grouping
        """
        if sample_rate > 1 or sample_rate <= 0:
            raise ValueError('invalid sample rate: %s' % sample_rate)
        if without_grouping and without_grouping not in \
                {'one_by_one', 'all_by_one'}:
            raise ValueError('unknown without grouping option')
        self._scale = scale
        self._rho = rho
        self._sigma = sigma
        self._sample_rate = sample_rate
        self._cluster_threshold = cluster_threshold
        self._random_state = random_state
        self._latest_windows = latest_windows
        # Sample point confidence calculation method
        self._sample_score_method = sample_score_method
        # The distance calculation method
        self._distance = distance
        # Retry parameters
        self._retry_limit = retry_limit
        # Maximum number of worker threads
        self._workers = workers
        # Downgrade an experiment
        self._without_grouping = without_grouping

    def reconstruct(self, data: np.ndarray, window: int = 20,
                    windows_per_cycle: int = 2, stride: int = 5):
        """Offline prediction of anomaly probability prediction
        in time windows of input data, multithreaded

        :param data: Input data
        :param window: Time window length (points)
        :param windows_per_cycle: Cycle length: In time window
        :param stride: Time window step
        """
        if windows_per_cycle < 1:
            raise ValueError('a cycle contains 1 window at least')
        cycle = windows_per_cycle * window
        groups = self._get_cycle_feature(data, cycle)

        reconstructed, retry_count = self._get_reconstructed_data(
            data, window, windows_per_cycle, groups, stride)
        return reconstructed, retry_count

    def predict(self, data: np.ndarray, reconstructed: np.ndarray,
                window: int, stride: int = 1):
        """Offline processing: Evaluate using parameters to obtain
        an anomaly score for each point

        :param data: Raw data
        :param reconstructed: The reconstructed data
        :param window: Data window length
        :param stride: Window step
        :return: Anomaly score for each point
        """
        if reconstructed.shape != data.shape:
            raise ValueError('shape mismatches')
        n, d = data.shape
        anomaly_score = np.zeros((n,))
        anomaly_score_weight = np.zeros((n,))
        wb = 0
        while True:
            we = min(n, wb + window)
            score = self._distance(data[wb:we], reconstructed[wb:we])
            for i in range(we - wb):
                w = i + wb
                weight = anomaly_score_weight[w]
                anomaly_score[w] = np.divide(
                    (anomaly_score[w] * weight + score), (weight + 1))
            anomaly_score_weight[wb:we] += 1
            if we >= n:
                break
            wb += stride
        return anomaly_score

    def _get_reconstructed_data(self, data: np.ndarray, window: int, windows_per_cycle: int,
                                groups: list, stride: int):
        """Offline prediction of anomaly probability prediction
        in time windows of input data, multithreaded

        :param data: Input data
        :param window: Time window length (points)
        :param windows_per_cycle: Cycle length: In time window
        :param groups: Grouping of each cycle
        :param stride: Time window step
        """
        n, d = data.shape
        reconstructed = np.zeros((n, d))
        reconstructing_weight = np.zeros((n,))
        needed_weight = np.zeros((n,))
        task_queue = Queue()
        result_queue = Queue()
        cycle = window * windows_per_cycle

        win_l = 0
        while True:
            win_r = min(n, win_l + window)
            task_queue.put((win_l, win_r, groups[win_l // cycle]))
            needed_weight[win_l:win_r] += 1
            if win_r >= n:
                break
            win_l += stride

        task_return_event = Event()
        finished = False

        def receive_result_thread():
            """
            The thread that accepts the result_queue result
            :return:
            """
            total_retries = 0
            while True:
                while result_queue.empty():
                    task_return_event.clear()
                    task_return_event.wait()
                    if finished:
                        result_queue.put(total_retries)
                        return
                wb, we, rec_window, retries, _ = result_queue.get()
                total_retries += retries
                for index in range(rec_window.shape[0]):
                    w = index + wb
                    weight = reconstructing_weight[w]
                    reconstructed[w, :] = divide(
                        reconstructed[w, :] * weight + rec_window[index],
                        weight + 1)
                reconstructing_weight[wb:we] += 1

        processes = []
        for i in range(self._workers):
            process = WindowReconstructProcess(
                data=data, task_queue=task_queue, result_queue=result_queue,
                cycle=cycle, latest_windows=self._latest_windows,
                sample_score_method=self._sample_score_method,
                sample_rate=self._sample_rate,
                scale=self._scale, rho=self._rho, sigma=self._sigma,
                random_state=self._random_state,
                retry_limit=self._retry_limit,
                task_return_event=task_return_event
            )
            process.start()
            processes.append(process)
        receiving_thread = Thread(target=receive_result_thread)
        receiving_thread.start()
        for each in processes:
            each.join()
        finished = True
        task_return_event.set()
        receiving_thread.join()

        mismatch_weights = []
        for i in range(n):
            if reconstructing_weight[i] != needed_weight[i]:
                mismatch_weights.append('%d' % i)
        if mismatch_weights:
            logger.error('BUG empty weight')
        return reconstructed, result_queue.get()

    def _get_cycle_feature(self, data: np.ndarray, cycle: int):
        """After dividing the data into cycles, the grouping of each cycle is calculated
        :param data: data
        :param cycle: Cycle length
        :return: Group results
        """
        n, d = data.shape
        cycle_groups = []
        group_index = 0
        task_queue = Queue()
        result_queue = Queue()
        cb = 0
        while cb < n:
            ce = min(n, cb + cycle)
            if group_index == 0 and not self._without_grouping:
                init_group = []
                for i in range(d):
                    init_group.append([i])
                cycle_groups.append(init_group)
            elif group_index == 0:
                init_group = []
                cycle_groups.append(init_group)
            else:
                cycle_groups.append([])
                if not self._without_grouping:
                    task_queue.put((group_index, data[cb:ce]))
            group_index += 1
            cb += cycle
        if self._without_grouping and self._without_grouping == 'one_by_one':
            for each in cycle_groups:
                for i in range(d):
                    each.append([i])
        # One group for all KPIs
        elif self._without_grouping and self._without_grouping == 'all_by_one':
            all_in_group = []
            for i in range(d):
                all_in_group.append(i)
            for each in cycle_groups:
                each.append(all_in_group)
        elif not self._without_grouping:
            processes = []
            for i in range(min(len(cycle_groups), self._workers)):
                process = CycleFeatureProcess(
                    task_queue, result_queue, self._cluster_threshold)
                process.start()
                processes.append(process)
            for process in processes:
                process.join()
            while not result_queue.empty():
                group_index, group = result_queue.get()
                cycle_groups[group_index] = group
        return cycle_groups
