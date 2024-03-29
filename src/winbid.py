import logging
import numpy as np
import pandas as pd
import scipy.stats as sp
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def cdf_from_samples(samples: np.ndarray) -> interp1d:
    """ Reconstruct the data distribution (its CDF) from data samples

    Parameters
    ----------
    samples: data samples (1d array)

    Return
    ------
    Cumulative distribution function (cdf) of the data generation distribution
    """

    assert samples.ndim == 1, "Provide a one-dimensional array of samples points"

    df = pd.DataFrame()
    df['x'] = np.sort(samples)
    df['y'] = (np.arange(len(samples)) + 1) / len(samples)
    df = df.groupby('x', as_index=False).agg(y=('y', max))

    return interp1d(df['x'], df['y'], bounds_error=False, fill_value=(0, 1))


def aucwin_prob_from_samples(
        samples_groups: list[np.ndarray],
        round_decimals: int = 2
):
    """ Use the win-prices from different auction groups to create a map that
    maps every bid to a win probability for every auction group

    Parameters
    ----------
    samples_groups: groups of samples from different distributions
    round_decimals: reduce the resolution of all winning bids to construct maps
        that use interpolations with less supporting nodes
    """

    # Probability to win an auction from the CDF of the win-price distribution
    cdfs = [cdf_from_samples(samples=samples.round(round_decimals))
            for samples in samples_groups]

    def p_aucwin_fn(x: np.ndarray):
        """ Probabilities to win different auctions by placing the bid x """

        assert x.ndim == 1

        return np.array([cdf(x_) for cdf, x_ in zip(cdfs, x)])

    return p_aucwin_fn


def aucwin_prob_analytical(dists: list[dict]):
    """ Create a map that maps every bid to a win probability for every
    auction group

    Parameters
    ----------
    dists: list of distribution definitions;
        dicts with keys `name` and `params`
    """

    # Probability to win an auction from the CDF of the win-price distribution
    cdfs = []
    for dsit in dists:
        winprice_dist = getattr(sp, dsit['name'])(**dsit['params'])
        cdfs.append(winprice_dist.cdf)

    def p_aucwin_fn(x: np.ndarray):
        """ Probabilities to win different auctions by placing the bid x """

        assert x.ndim == 1

        return np.array([cdf(x_) for cdf, x_ in zip(cdfs, x)])

    return p_aucwin_fn


def generate_winbid_samples(
        name: str,
        params: dict,
        samples: int = 10_000
) -> np.ndarray:
    """ Generate samples from a win-bid distribution

    Parameters
    ----------
    name: distribution from scipy.stats (for example 'expon')
    params: distribution parameters
    samples: desired number of samples
    """

    winbid_dist = getattr(sp, name)(**params)
    samples = winbid_dist.rvs(size=samples)

    return samples
