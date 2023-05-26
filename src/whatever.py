import logging
import numpy as np
from functools import partial
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def f(x: np.ndarray, p_win_fn, n: np.ndarray):
    """ Expected amount spent for won auctions """

    assert x.ndim == p_win_fn(x).ndim == n.ndim == 1
    assert len(x) == len(p_win_fn(x)) == len(n)

    return (n * x * p_win_fn(x)).sum()


def g(x: np.ndarray, p_win_fn, p_ctr: np.ndarray, n: np.ndarray, n_click: float):
    """ Difference btw the desired and expected number of clicks based on a
    bidding strategy

    Parameters
    ----------
    x: bids for every auction type
    p_win_fn: fn that maps x to win probabilities for every auction type
    p_ctr: probability that a user will click on the displayed ad
    n: expected number of auctions for every auction type
    n_click: expected number of clicks
    """

    assert x.ndim == p_win_fn(x).ndim == p_ctr.ndim == n.ndim == 1
    assert len(x) == len(p_win_fn(x)) == len(n) == len(p_ctr)

    return (n * p_ctr * p_win_fn(x)).sum() - n_click


def optimize(
        x0: np.ndarray,
        p_win_fn,
        p_ctr: np.ndarray,
        n: np.ndarray,
        n_click: float
):
    """ Find optimal bids for the different auction types """

    assert x0.ndim == p_win_fn(x0).ndim == p_ctr.ndim == n.ndim == 1
    assert len(x0) == len(p_win_fn(x0)) == len(p_ctr) == len(n)

    logger.info(f'Find optimal bids for {len(n)} auction types')

    # Allowed ranges for the bids: (0, infinity)
    bounds = [(0, None) for _ in n]

    # Boundary condition: expected number of clicks = nc
    args_g = {'p_win_fn': p_win_fn, 'p_ctr': p_ctr, 'n': n, 'n_click': n_click}
    cons = ({'type': 'eq', 'fun': lambda x: g(x=x, **args_g)})

    args = {'method': 'trust-constr', 'bounds': bounds, 'constraints': cons}
    res = minimize(fun=partial(f, p_win_fn=p_win_fn, n=n), x0=x0, **args)

    return res
