import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from collections import namedtuple
from optimizer import AuctionGroups, Optimizer
from winbid import aucwin_prob_analytical


class TestOptimizer:
    """ Test the Optimizer for the case of having a single auction group """

    Example = namedtuple('example', ['n_click', 'p_ctr', 'n', 'alpha'])

    ex01 = Example(n_click=100,
                   p_ctr=np.array([.01]),
                   n=np.array([30_000]),
                   alpha=1)

    ex02 = Example(n_click=200,
                   p_ctr=np.array([.01]),
                   n=np.array([30_000]),
                   alpha=1)

    ex03 = Example(n_click=200,
                   p_ctr=np.array([.01]),
                   n=np.array([30_000]),
                   alpha=2)

    def optimal_bid(
            self,
            n: np.ndarray,
            p_ctr: np.ndarray,
            n_click: float | int,
            alpha: float
    ) -> np.ndarray:
        """ Analytical solution for the optimal bid """

        return np.log(n * p_ctr / (n * p_ctr - n_click)) / alpha

    def setup_method(self):
        self.tol = 1e-3

    @pytest.mark.parametrize('ex', [ex01, ex02, ex03])
    def test_optimizer(self, ex: Example):
        """ Test the optimizer """

        winbid_dists = [{"name": "expon", "params": {'scale': 1. / ex.alpha}}]
        p_aucwin_fn = aucwin_prob_analytical(dists=winbid_dists)

        groups = AuctionGroups(p_ctr=ex.p_ctr, n=ex.n, p_aucwin_fn=p_aucwin_fn)
        optimizer = Optimizer(groups=groups, n_click=ex.n_click, max_bid=10.)
        bid = optimizer.optimal_bid_strategy()

        bid_expected = self.optimal_bid(
            n=ex.n, p_ctr=ex.p_ctr, n_click=ex.n_click, alpha=ex.alpha)

        assert_almost_equal(bid, bid_expected, decimal=3)
