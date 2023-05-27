import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize, Bounds
from typing import Callable
from pydantic import BaseModel, root_validator

logger = logging.getLogger(__name__)


class AuctionGroups(BaseModel):
    """ Ad-auction groups

    :param p_ctr: click-through probabilities per auction group
    :param n: expected number of auctions per auction group
    :param p_aucwin_fn: function that maps bids for the different auction
        groups to probabilities to win an auction
    """

    p_ctr: np.ndarray
    n: np.ndarray
    p_aucwin_fn: Callable[[np.ndarray], np.ndarray]

    class Config:  # noqa
        arbitrary_types_allowed = True

    @root_validator
    def check_dimensions_match(cls, v):  # noqa
        p_ctr, n, p_aucwin_fn = v.get('p_ctr'), v.get('n'), v.get('p_aucwin_fn')

        x = np.zeros_like(n)

        assert p_ctr.ndim == n.ndim == p_aucwin_fn(x).ndim == 1

        assert len(p_ctr) == len(n) == len(p_aucwin_fn(x))

        assert ((0 <= p_ctr) & (p_ctr <= 1)).all(), \
            "Click-through probabilities should be btw 0 and 1"

        assert (n >= 0).all(), \
            "Expected number of ad-auctions should be non-negative"

        return v

    @property
    def n_groups(self) -> int:
        """ Number of ad-auction groups """
        return len(self.n)

    def plot(
            self,
            x_range: tuple[float, float] = (0., 5.),
            const: np.ndarray = None,
            const_label: str = None
    ):
        """ Plot winbid probabilities and distributions for all auction groups
        """

        dim = self.n_groups

        x_range = np.linspace(*x_range, 100)

        # shape: (100, dim)
        xs = x_range.reshape(-1, 1) * np.ones(dim).reshape(1, -1)
        ys = np.stack([self.p_aucwin_fn(x) for x in xs])

        fig = plt.figure(figsize=(10, 2 * dim))

        for d in range(dim):
            ax = plt.subplot(dim, 2, 2 * d + 1)
            ax.plot(x_range, ys[:, d])
            if const is not None:
                label = f'{const_label} (p_ctr={self.p_ctr[d]:.3f})'
                ax.axvline(const[d], c='k', linestyle='--', label=label)
            ax.legend(loc='lower right')
            if d == 0:
                ax.set_title('Auction win probability')

            ax = plt.subplot(dim, 2, 2 * d + 2)
            ax.plot(x_range[:-1], np.diff(ys[:, d]))
            if const is not None:
                label = f'{const_label} (p_ctr={self.p_ctr[d]:.3f})'
                ax.axvline(const[d], c='k', linestyle='--', label=label)
            ax.legend(loc='upper right')
            if d == 0:
                ax.set_title('Win-bid distribution')

        plt.tight_layout()

        return fig


class Optimizer:
    """ Find optimal bids for a particular set of ad-auction gr

    :param groups: ad-auction gr
    :param n_click: desired number of ad clicks
    :param max_bid: maximum possible bid that we are willing to place
    """

    def __init__(
            self,
            groups: AuctionGroups,
            n_click: float | int,
            max_bid: float = 10.
    ):
        self.gr = groups
        self.n_click = n_click
        self.max_bid = max_bid
        self._feasibility_check()

    # TODO: take into account the maximum possible bid that we are willing to place
    def _feasibility_check(self):
        """ Feasibility check of the optimization problem """

        # Expected number of clicks if we win every auction
        nc_max = (self.gr.n * self.gr.p_ctr).sum()

        assert self.n_click < nc_max, ("The desired amount of clicks is lower "
                                       "than the expected number of clicks if "
                                       "we win every auction.")

    def _initial_value_optimization(self):
        """ Get the initial value for the optimization algorithm

        Pick a value where the first derivative of the win-bid probability is
        nearly maximal, i.e. the most common win-bid.
        """

        dim = self.gr.n_groups

        search_range = np.linspace(0, self.max_bid, 100)

        # shape: (100, dim)
        xs = np.ones((1, dim)) * search_range.reshape(-1, 1)

        # shape: (100, dim)
        ys = np.stack([self.gr.p_aucwin_fn(x) for x in xs])

        # shape: (99, dim)
        ys_diff = np.diff(ys, axis=0)

        # shape: (dim,)
        indices = ys_diff.argmax(axis=0)

        x0 = search_range[indices]

        # jitter
        x0 += np.random.rand() * .05
        x0 = x0.clip(0, self.max_bid)

        return x0

    def spending(self, x: np.ndarray):
        """ Expected amount spent for won auctions

        Parameters
        ----------
        x: bids for every auction type
        """

        return (self.gr.n * x * self.gr.p_aucwin_fn(x)).sum()

    def clicks(self, x: np.ndarray):
        """ Expected number of clicks based on a bidding strategy

        Parameters
        ----------
        x: bids for every auction type
        """

        n = (self.gr.n * self.gr.p_ctr * self.gr.p_aucwin_fn(x)).sum()

        return n

    def optimize(self):
        """ Find optimal bids for the different auction types """

        logger.info(
            f'Find the optimal bids for {len(self.gr.n)} auction groups')

        dim = len(self.gr.p_ctr)

        # Lower and upper bid bounds: 0 <= bid <= max_bid
        lb = np.zeros(dim)
        ub = np.ones(dim) * self.max_bid

        # Rescale trainable params such that they are of magnitude ~ 1
        scale = self.max_bid
        bounds = Bounds(lb=lb / scale,
                        ub=ub / scale, keep_feasible=True)

        # Initial guess for the optimal bids
        x0 = self._initial_value_optimization() / scale

        # Boundary condition: expected = desired number of clicks
        cons = ({'type': 'eq',
                 'fun': lambda x: self.clicks(scale * x) / self.n_click - 1})

        res = minimize(
            fun=lambda x: self.spending(scale * x) / self.spending(ub),
            x0=x0,
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 2_000},
            tol=1e-6)

        # Map the results to the original scale
        res.x *= scale
        res.clicks = self.clicks(res.x)
        res.spending = self.spending(res.x)

        return res

    def optimal_bid_strategy(self, trials: int = 10):
        """ Find the optimal bid strategy """

        results = [self.optimize() for _ in range(trials)]
        results = [res for res in results if res.success]

        summary = pd.DataFrame()
        summary['results'] = [res.x for res in results]
        summary['clicks'] = [res.clicks for res in results]
        summary['spending'] = [res.spending for res in results]

        for i in range(self.gr.n_groups):
            summary[f'b{i}'] = [res.x[i] for res in results]

        summary = summary.sort_values(by=['spending'])

        logger.info(f'Summary: {summary.drop(["results"], axis=1).round(2)}')

        assert len(summary) > 0, "No successful optimization"

        bid = summary['results'].values[0]

        return bid
