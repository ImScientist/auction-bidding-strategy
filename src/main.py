import json
import click
import logging
import numpy as np

from optimizer import AuctionGroups, Optimizer
from winbid import (
    generate_winbid_samples,
    aucwin_prob_from_samples,
    aucwin_prob_analytical)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    '--winbid-dist',
    type=str,
    default=
    '['
    '{"name": "exponpow", "params": {"b": 2}, "samples": 10000},'
    '{"name": "exponpow", "params": {"b": 2}, "samples": 10000},'
    '{"name": "exponpow", "params": {"b": 4}, "samples": 10000},'
    '{"name": "exponpow", "params": {"b": 4}, "samples": 10000},'
    '{"name": "exponpow", "params": {"b": 6}, "samples": 10000},'
    '{"name": "exponpow", "params": {"b": 6}, "samples": 10000}'
    ']',
    help='Combinations of CTR probabilities, expected number of incoming '
         'requests, and auction winning bid distributions')
@click.option(
    '--save-path',
    type=str,
    default='data.json',
    help='Path to store the generated data')
def generate_data(winbid_dist, save_path):
    """ Generate win-bid samples for different ad-auction groups """

    winbid_dist = json.loads(winbid_dist)

    logger.info(f'Generate data for win-bid distributions...')

    data = [{'winbid_samples': generate_winbid_samples(**dist).tolist()}
            for dist in winbid_dist]

    logger.info(f'Store data in {save_path}.')
    with open(save_path, "w") as f:
        json.dump(data, f)


@cli.command()
@click.option(
    '--n-click',
    type=float,
    default=100.,
    help='Desired number of user ad clicks')
@click.option(
    '--p-ctr',
    type=str,
    default='0.005, 0.010, 0.005, 0.010, 0.005, 0.010',
    help='Click-through probabilities per auction group')
@click.option(
    '--n',
    type=str,
    default='100_000, 50_000, 100_000, 20_000, 200_000, 60_000',
    help='Expected number of auctions per auction group')
@click.option(
    '--winbid-dist',
    type=str,
    default=
    '['
    '{"name": "exponpow", "params": {"b": 2}},'
    '{"name": "exponpow", "params": {"b": 2}},'
    '{"name": "exponpow", "params": {"b": 4}},'
    '{"name": "exponpow", "params": {"b": 4}},'
    '{"name": "exponpow", "params": {"b": 6}},'
    '{"name": "exponpow", "params": {"b": 6}}'
    ']',
    help=
    'Winning bid distributions per auction group. Depending on the '
    '`analytical` flag, it is either a json string with the definitions of '
    'the winbid distributions (flag used) or a path where samples for every '
    'auction group are stored (no flag).')
@click.option(
    '--winbid-dist-analytical', '-a',
    is_flag=True,
    help="Whether we define win-bid dists diretly by analytical functions or "
         "we infer them by looking at samples from the distributions.")
def optimize(n_click, p_ctr, n, winbid_dist, winbid_dist_analytical):
    """ Use empirical data for different auction types to construct an optimal
    bidding strategy """

    if winbid_dist_analytical:
        logger.info('Use explicitly defined win-bid distributions')

        # Fn that maps bids for different auction types to winning probabilities
        p_aucwin_fn = aucwin_prob_analytical(dists=json.loads(winbid_dist))
    else:
        logger.info(f'Load samples of the win-bid distributions from {winbid_dist}')

        with open(winbid_dist, "r") as f:
            data = json.load(f)

        # Fn that maps bids for different auction types to winning probabilities
        p_aucwin_fn = aucwin_prob_from_samples(
            samples_groups=[np.array(x['winbid_samples']) for x in data])

    # Click-through probabilities
    p_ctr = np.array([float(x) for x in p_ctr.split(',')])

    # Expected number of auctions per auction type
    n = np.array([float(x) for x in n.split(',')])

    groups = AuctionGroups(p_ctr=p_ctr, n=n, p_aucwin_fn=p_aucwin_fn)
    optimizer = Optimizer(groups=groups, n_click=n_click)
    bid = optimizer.optimal_bid_strategy()

    # Plot the optimal solution
    fig = groups.plot(x_range=(0, optimizer.max_bid), const=bid, const_label='bid')
    fig.savefig('solution.png', dpi=300, bbox_inches='tight')

    logger.info(f'Optimal bid: {bid.round(2)}')
    logger.info('Figure stored in solution.png')


if __name__ == "__main__":
    """ PYTHONPATH=src python src/main.py --help """
    cli()
