""" Infer probability distribution functions (pdfs) from data
or use predefined pdfs.
"""

import numpy as np
from functools import partial
import scipy.stats as sp
from scipy.interpolate import UnivariateSpline


def histogram_to_spline(data, bins=None):
    """ Create an approximation of the pdf from the observed data
    """
    if bins is None:
        bins = np.linspace(0, 10, num=501, endpoint=True)

    counts, boundaries = np.histogram(data, bins=bins)

    bins_mid = (bins[1:] + bins[:-1]) / 2
    spl = UnivariateSpline(bins_mid, counts, k=3, s=0)

    norm = spl.integral(bins[0], bins[-1])
    counts = counts / norm
    spl = UnivariateSpline(bins_mid, counts, k=3, s=0)

    return spl


def get_ps_pw_from_data(datasets, J):
    """ Get the the pdfs ps(), pw()

    :param datasets: list of I datasets
    :param J: number of different p_c bins
    :return:
    """
    ps_list = []
    pw_list = []

    for dataset in datasets:
        spl = histogram_to_spline(dataset)
        ps_list.append(spl)
        pw_list.append(partial(spl.integral, a=0))

    ps_list = np.tile(ps_list, J)
    pw_list = np.tile(pw_list, J)

    def ps(x):
        return np.array([ps_(x_) for ps_, x_ in zip(ps_list, x)])

    def ps_der(x):
        return np.array([ps_.derivative()(x_) for ps_, x_ in zip(ps_list, x)])

    def pw(x):
        return np.array([pw_(b=x_) for pw_, x_ in zip(pw_list, x)])

    return ps, ps_der, pw


def get_ps_pw_analytical(J: int, distribution_type: str, dist_params_list=None):
    """ Get the the pdfs ps(), pw()

    :param distribution_type: distribution from scipy.stats (for example 'expon')
    :param J: number of different p_c bins
    :param dist_params_list:
    :return:
    """

    ps_list = []
    pw_list = []

    for dist_params in dist_params_list:
        ps = getattr(sp, distribution_type)(**dist_params)
        pw = ps.cdf

        ps_list.append(ps)
        pw_list.append(pw)

    ps_list = np.tile(ps_list, J)
    pw_list = np.tile(pw_list, J)

    def ps(x):
        return np.array([ps_(x_) for ps_, x_ in zip(ps_list, x)])

    def pw(x):
        return np.array([pw_(x_) for pw_, x_ in zip(pw_list, x)])

    return ps, None, pw
