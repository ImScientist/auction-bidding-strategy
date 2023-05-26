import logging
import numpy as np
from scipy.optimize import minimize
from typing import Callable
from pydantic import BaseModel, root_validator

logger = logging.getLogger(__name__)


class AuctionGroups(BaseModel):
    """ Ad-auction groups

    :param p_ctr: click-through probabilities per auction group
    :param n: expected number of auctions per auction group
    :param p_win_fn: function that maps bids for the different auction groups
        to winning bid probabilities
    """

    p_ctr: np.ndarray
    n: np.ndarray
    p_win_fn: Callable[[np.ndarray], np.ndarray]

    class Config:  # noqa
        arbitrary_types_allowed = True

    @root_validator
    def check_dimensions_match(cls, v):  # noqa
        p_ctr, n, p_win_fn = v.get('p_ctr'), v.get('n'), v.get('p_win_fn')

        x = np.zeros_like(n)

        assert p_ctr.ndim == n.ndim == p_win_fn(x).ndim == 1

        assert len(p_ctr) == len(n) == len(p_win_fn(x))

        assert ((0 <= p_ctr) & (p_ctr <= 1)).all(), \
            "Click-through probabilities should be btw 0 and 1"

        assert (n >= 0).all(), \
            "Expected number of ad-auctions should be non-negative"

        return v


class Optimizer:
    """ Find optimal bids for a particular set of ad-auction groups

    :param groups: ad-auction groups
    :param n_click: Desired number of ad clicks
    """

    def __init__(self, groups: AuctionGroups, n_click: float | int):
        self.groups = groups
        self.n_click = n_click
        self._feasibility_check()

    def _feasibility_check(self):
        """ Feasibility check of the optimization problem """

        # Expected number of clicks if we win every auction
        nc_max = (self.groups.n * self.groups.p_ctr).sum()

        assert self.n_click < nc_max, ("The desired amount of clicks is lower "
                                       "than the expected number of clicks if "
                                       "we win every auction.")

    def f(self, x: np.ndarray):
        """ Expected amount spent for won auctions """

        return (self.groups.n * x * self.groups.p_win_fn(x)).sum()

    def g(self, x: np.ndarray):
        """ Difference btw the desired and expected number of clicks based on a
        bidding strategy

        Parameters
        ----------
        x: bids for every auction type
        """

        n = (self.groups.n * self.groups.p_ctr * self.groups.p_win_fn(x)).sum()

        return n - self.n_click

    def optimize(self, x0: np.ndarray = None):
        """ Find optimal bids for the different auction types """

        logger.info(f'Find the optimal bids for {len(self.groups.n)} auction '
                    f'groups')

        if x0 is None:
            x0 = np.ones_like(self.groups.p_ctr)

        # Allowed ranges for the bids: (0, infinity)
        bounds = [(0, None) for _ in self.groups.n]

        # Boundary condition: expected number of clicks = nc
        cons = ({'type': 'eq', 'fun': self.g})

        args = {'method': 'trust-constr', 'bounds': bounds, 'constraints': cons}
        res = minimize(fun=self.f, x0=x0, **args)

        return res
