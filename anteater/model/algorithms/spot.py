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

import tqdm
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from math import log, floor

from anteater.utils.log import logger


class Base:
    def __init__(self):
        self.proba = None
        self.data = None
        self.init_data = None
        self.n = 0
        self.nt = 0
        self.init_threshold = None

    @staticmethod
    def u(s):
        return 1 + np.log(s).mean()

    @staticmethod
    def v(s):
        return np.mean(1 / s)

    @classmethod
    def roots_finder(cls, fun, jac, bounds, npoints, method):
        """
        Find possible roots of a scalar function

        Parameters
        ----------
        fun : function
            scalar function
        jac : function
            first order derivative of the function
        bounds : tuple
            (min,max) interval for the roots search
        npoints : int
            maximum number of roots to output
        method : str
            'regular' : regular sample of the search interval, 'random' : uniform sample of the search interval

        Returns
        ----------
        numpy.array
            possible roots of the function
        """
        if method == 'regular':
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            x_0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == 'random':
            x_0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def obj_fun(xs, f, jac):
            g = 0
            j = np.zeros(xs.shape)
            i = 0
            for x in xs:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j

        opt = minimize(lambda xs: obj_fun(xs, fun, jac), x_0,
                       method='L-BFGS-B',
                       jac=True, bounds=[bounds] * len(x_0))

        xs = opt.x
        np.round(xs, decimals=5)
        return np.unique(xs)

    @classmethod
    def log_likelihood(cls, y, gamma, sigma):
        """
        Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)

        Parameters
        ----------
        y : numpy.array
            observations
        gamma : float
            GPD index parameter
        sigma : float
            GPD scale parameter (>0)

        Returns
        ----------
        float
            log-likelihood of the sample y to be drawn from a GPD(γ,σ,μ=0)
        """
        n = y.size
        if gamma != 0:
            tau = gamma / sigma
            l = -n * log(sigma) - (1 + (1 / gamma)) * \
                (np.log(1 + tau * y)).sum()
        else:
            l = n * (1 + log(y.mean()))
        return l

    @classmethod
    def w(cls, y, t):
        s = 1 + t * y
        us = cls.u(s)
        vs = cls.v(s)
        return us * vs - 1

    @classmethod
    def jac_w(cls, y, t):
        s = 1 + t * y
        us = cls.u(s)
        vs = cls.v(s)
        jac_us = (1 / t) * (1 - vs)
        jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
        return us * jac_vs + vs * jac_us

    def fit(self, init_data, data):
        """
        Import data to SPOT object

        Parameters
        ----------
        init_data : list, numpy.array or pandas.Series
            initial batch to calibrate the algorithm

        data : numpy.array
            data for the run (list, np.array or pd.series)

        """
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = data.values
        else:
            logger.error('This data format (%s) is not supported', type(data))
            return

        if isinstance(init_data, list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data, pd.Series):
            self.init_data = init_data.values
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) & (init_data < 1) & (init_data > 0):
            r = int(init_data * data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            logger.error('The initial data cannot be set')
            return

    def add(self, data):
        """
        This function allows to append data to the already fitted data

        Parameters
        ----------
        data : list, numpy.array, pandas.Series
            data to append
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            data = data
        elif isinstance(data, pd.Series):
            data = data.values
        else:
            logger.error('This data format (%s) is not supported', type(data))
            return

        self.data = np.append(self.data, data)
        return

    def _quantile(self, gamma, sigma):
        """
        Compute the quantile at level 1-q

        Parameters
        ----------
        gamma : float
            GPD parameter
        sigma : float
            GPD parameter

        Returns
        ----------
        float
            quantile at level 1-q for the GPD(γ,σ,μ=0)
        """
        r = self.n * self.proba / self.nt
        if gamma != 0:
            return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            return self.init_threshold - sigma * log(r)


class Spot(Base):
    """
    This class allows to run SPOT algorithm on univariate dataset (upper-bound)

    Attributes
    ----------
    proba : float
        Detection level (risk), chosen by the user

    extreme_quantile : float
        current threshold (bound between normal and abnormal events)

    data : numpy.array
        stream

    init_data : numpy.array
        initial batch of observations (for the calibration/initialization step)

    init_threshold : float
        initial threshold computed during the calibration step

    peaks : numpy.array
        array of peaks (excesses above the initial threshold)

    n : int
        number of observed values

    nt : int
        number of observed peaks
    """

    def __init__(self, q=1e-4):
        """
        Constructor

        Parameters
        ----------
        q
            Detection level (risk)

        Returns
        ----------
        SPOT object
        """
        super().__init__()
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.nt = 0

    def initialize(self, level=0.98, min_extrema=False, verbose=True):
        """
        Run the calibration (initialization) step

        Parameters
        ----------
        level : float
            (default 0.98) Probability associated with the initial threshold t
        verbose : bool
            (default = True) If True, gives details about the batch initialization
        verbose: bool
            (default True) If True, prints log
        min_extrema bool
            (default False) If True, find min extrema instead of max extrema
        """
        if min_extrema:
            self.init_data = -self.init_data
            self.data = -self.data
            level = 1 - level

        level = level - floor(level)

        n_init = self.init_data.size

        sort_data_s = np.sort(self.init_data)  # we sort X to get the empirical quantile
        # t is fixed for the whole algorithm
        self.init_threshold = sort_data_s[int(level * n_init)]

        # initial peaks
        self.peaks = self.init_data[self.init_data >
                                    self.init_threshold] - self.init_threshold
        self.nt = self.peaks.size
        self.n = n_init

        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)

        return

    def run(self, with_alarm=True, dynamic=True):
        """
        Run SPOT on the stream

        Parameters
        ----------
        with_alarm : bool
            (default = True) If False, SPOT will adapt the threshold assuming \
            there is no abnormal values


        Returns
        ----------
        dict
            keys : 'thresholds' and 'alarms'

            'thresholds' contains the extreme quantiles and 'alarms' contains \
            the indexes of the values which have triggered alarms

        """
        if self.n > self.init_data.size:
            logger.warning('The algorithm seems to have already been run, you \
            should initialize before running again')
            return {}

        # list of the thresholds
        th = []
        alarm = []
        # Loop over the stream
        for i in tqdm.tqdm(range(self.data.size)):

            if not dynamic:
                if self.data[i] > self.init_threshold and with_alarm:
                    self.extreme_quantile = self.init_threshold
                    alarm.append(i)
            else:
                # If the observed value exceeds the current threshold (alarm case)
                if self.data[i] > self.extreme_quantile and with_alarm:
                    # if we want to alarm, we put it in the alarm list
                    alarm.append(i)
                    # otherwise we add it in the peaks
                elif self.data[i] > self.extreme_quantile and not with_alarm:
                    self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                    self.nt += 1
                    self.n += 1
                    # and we update the thresholds
                    g, s, l = self._grimshaw()
                    self.extreme_quantile = self._quantile(g, s)

                # case where the value exceeds the initial threshold but not the alarm ones
                elif self.data[i] > self.init_threshold:
                    # we add it in the peaks
                    self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                    self.nt += 1
                    self.n += 1
                    # and we update the thresholds

                    g, s, l = self._grimshaw()
                    self.extreme_quantile = self._quantile(g, s)
                else:
                    self.n += 1

            th.append(self.extreme_quantile)  # thresholds record

        return {'thresholds': th, 'alarms': alarm}

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        """
        Compute the GPD parameters estimation with the Grimshaw's trick

        Parameters
        ----------
        epsilon : float
            numerical parameter to perform (default : 1e-8)
        n_points : int
            maximum number of candidates for maximum likelihood (default : 10)

        Returns
        ----------
        gamma_best,sigma_best,ll_best
            gamma estimates, sigma estimates and corresponding log-likelihood
        """

        y_min = self.peaks.min()
        y_max = self.peaks.max()
        y_mean = self.peaks.mean()

        a = -1 / y_max
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        if y_mean == y_min:
            y_min = 0.999 * y_min
        b = 2 * (y_mean - y_min) / (y_mean * y_min)
        c = 2 * (y_mean - y_min) / (y_min ** 2)

        # We look for possible roots
        left_zeros = Spot.roots_finder(lambda t: Spot.w(self.peaks, t),
                                       lambda t: Spot.jac_w(self.peaks, t),
                                       (a + epsilon, -epsilon),
                                       n_points, 'regular')

        right_zeros = Spot.roots_finder(lambda t: Spot.w(self.peaks, t),
                                        lambda t: Spot.jac_w(self.peaks, t),
                                        (b, c),
                                        n_points, 'regular')

        # all the possible roots
        zeros = np.concatenate((left_zeros, right_zeros))

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = y_mean
        ll_best = Spot.log_likelihood(self.peaks, gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = Spot.u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = Spot.log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best


"""
================================= WITH DRIFT ==================================
"""


def back_mean(x, d):
    m = []
    w = x[:d].sum()
    m.append(w / d)
    for i in range(d, len(x)):
        w = w - x[i - d] + x[i]
        m.append(w / d)
    return np.array(m)


class DSpot(Base):
    """
    This class allows to run DSPOT algorithm on univariate dataset (upper-bound)

    Attributes
    ----------
    proba : float
        Detection level (risk), chosen by the user

    depth : int
        Number of observations to compute the moving average

    extreme_quantile : float
        current threshold (bound between normal and abnormal events)

    data : numpy.array
        stream

    init_data : numpy.array
        initial batch of observations (for the calibration/initialization step)

    init_threshold : float
        initial threshold computed during the calibration step

    peaks : numpy.array
        array of peaks (excesses above the initial threshold)

    n : int
        number of observed values

    nt : int
        number of observed peaks
    """

    def __init__(self, q, depth):
        super().__init__()
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.nt = 0
        self.depth = depth

    def initialize(self, verbose=True):
        """
        Run the calibration (initialization) step

        Parameters
        ----------
        verbose : bool
            (default = True) If True, gives details about the batch initialization
        """
        n_init = self.init_data.size - self.depth

        back_mean_m = back_mean(self.init_data, self.depth)
        data_t = self.init_data[self.depth:] - back_mean_m[:-1]  # new variable

        sort_data_s = np.sort(data_t)  # we sort X to get the empirical quantile
        # t is fixed for the whole algorithm
        self.init_threshold = sort_data_s[int(0.98 * n_init)]

        # initial peaks
        self.peaks = data_t[data_t > self.init_threshold] - self.init_threshold
        self.nt = self.peaks.size
        self.n = n_init

        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)

        return

    def run(self, with_alarm=True):
        """
        Run biSPOT on the stream

        Parameters
        ----------
        with_alarm : bool
            (default = True) If False, SPOT will adapt the threshold assuming \
            there is no abnormal values


        Returns
        ----------
        dict
            keys : 'upper_thresholds', 'lower_thresholds' and 'alarms'

            '***-thresholds' contains the extreme quantiles and 'alarms' contains \
            the indexes of the values which have triggered alarms

        """
        if (self.n > self.init_data.size):
            logger.warning('The algorithm seems to have already been run, you \
            should initialize before running again')
            return {}

        # actual normal window
        norm_w = self.init_data[-self.depth:]

        # list of the thresholds
        th = []
        alarm = []
        # Loop over the stream
        for i in tqdm.tqdm(range(self.data.size)):
            norm_w_mean = norm_w.mean()
            # If the observed value exceeds the current threshold (alarm case)
            if (self.data[i] - norm_w_mean) > self.extreme_quantile and with_alarm:
                # if we want to alarm, we put it in the alarm list
                alarm.append(i)
                # otherwise we add it in the peaks
            elif (self.data[i] - norm_w_mean) > self.extreme_quantile and not with_alarm:
                self.peaks = np.append(self.peaks, self.data[i] - norm_w_mean - self.init_threshold)
                self.nt += 1
                self.n += 1
                # and we update the thresholds
                g, s, l = self._grimshaw()
                self.extreme_quantile = self._quantile(g, s)  # + norm_w_mean
                norm_w = np.append(norm_w[1:], self.data[i])

            # case where the value exceeds the initial threshold but not the alarm ones
            elif (self.data[i] - norm_w_mean) > self.init_threshold:
                # we add it in the peaks
                self.peaks = np.append(
                    self.peaks, self.data[i] - norm_w_mean - self.init_threshold)
                self.nt += 1
                self.n += 1
                # and we update the thresholds

                g, s, l = self._grimshaw()
                self.extreme_quantile = self._quantile(g, s)  # + norm_w_mean
                norm_w = np.append(norm_w[1:], self.data[i])
            else:
                self.n += 1
                norm_w = np.append(norm_w[1:], self.data[i])

            th.append(self.extreme_quantile + norm_w_mean)  # thresholds record

        return {'thresholds': th, 'alarms': alarm}

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        """
        Compute the GPD parameters estimation with the Grimshaw's trick

        Parameters
        ----------
        epsilon : float
            numerical parameter to perform (default : 1e-8)
        n_points : int
            maximum number of candidates for maximum likelihood (default : 10)

        Returns
        ----------
        gamma_best,sigma_best,ll_best
            gamma estimates, sigma estimates and corresponding log-likelihood
        """

        y_min = self.peaks.min()
        y_max = self.peaks.max()
        y_mean = self.peaks.mean()

        a = -1 / y_max
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (y_mean - y_min) / (y_mean * y_min)
        c = 2 * (y_mean - y_min) / (y_min ** 2)

        # We look for possible roots
        left_zeros = DSpot.roots_finder(lambda t: DSpot.w(self.peaks, t),
                                        lambda t: DSpot.jac_w(self.peaks, t),
                                        (a + epsilon, -epsilon),
                                        n_points, 'regular')

        right_zeros = DSpot.roots_finder(lambda t: DSpot.w(self.peaks, t),
                                         lambda t: DSpot.jac_w(self.peaks, t),
                                         (b, c),
                                         n_points, 'regular')

        # all the possible roots
        zeros = np.concatenate((left_zeros, right_zeros))

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = y_mean
        ll_best = DSpot.log_likelihood(self.peaks, gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = DSpot.u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = DSpot.log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best
