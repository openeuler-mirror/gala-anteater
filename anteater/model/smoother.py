import numpy as np
from scipy.signal import savgol_filter


def conv_smooth(y, box_pts):
    """Apply a convolution smoothing to an array"""
    box = np.divide(np.ones(box_pts), box_pts)
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth


def savgol_smooth(y, *args, **kwargs):
    """Apply a Savitzky-Golay filter to an array"""
    return savgol_filter(y, *args, *kwargs)
