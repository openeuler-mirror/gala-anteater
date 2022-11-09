import numpy as np

from anteater.model.smoother import conv_smooth


def slope(y, windows_length):
    """Calculates point slope in an array"""
    if len(y) <= windows_length:
        raise ValueError('point_slope: the length of array should'
                         f'greater than window_length : f{windows_length}.')

    return np.divide(y[windows_length:], y[: -windows_length])


def smooth_slope(time_series, windows_length):
    val = conv_smooth(time_series.to_df(), box_pts=13)
    val = slope(val, windows_length=13)
    return val[-windows_length:]
