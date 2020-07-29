import numpy as np
from functools import partial
from scipy.optimize import minimize


def f(x, pw, N):
    return (N * x * pw(x)).sum()


def f_der(x, ps, pw, N):
    return N * (pw(x) + x * ps(x))


def f_hess(x, ps, ps_der, N):
    return np.diag(N * (2 * ps(x) + x * ps_der(x)))


def g(x, pw, N, pc, Nc):
    """
    Nc: expected number of clicks
    pc: probability that a user will click on the displayed ad
    """
    return (N * pc * pw(x)).sum() - Nc


def optimize(
        x0, pw, N, pc, Nc
):
    # allowed range for the bids (0, infinity)
    bnds = [(0, None) for _ in N]

    # boundary condition: expected number of clicks = Nc
    cons = ({'type': 'eq', 'fun': lambda x: g(x, pw, N, pc, Nc)})

    res = minimize(
        fun=partial(f, pw=pw, N=N),
        x0=x0,
        # jac=partial(f_der, ps=ps, pw=pw, N=N),
        # hess=partial(f_hess, ps=ps, ps_der=ps_der, N=N),
        method='trust-constr', bounds=bnds, constraints=cons)

    return res
