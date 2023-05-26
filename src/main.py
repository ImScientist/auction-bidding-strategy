import json
import click
import logging
import numpy as np
import pandas as pd

from whatever import (
    generate_winbid_samples,
    winbid_prob_from_samples,
    winbid_prob_analytical,
    optimize,
    AuctionGroups)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# TODO:
def get_summary(res, dist_params_list, pc, N):
    df = pd.DataFrame(index=pd.Index(np.arange(len(N))))
    df['x'] = res.x
    df['dist_params'] = dist_params_list
    df['dist_params'] = df['dist_params'].map(str)
    df['N_group'] = N
    df['p_ctr'] = pc
    return df


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    '--auction-groups',
    type=str,
    default=
    '['
    '{"p_ctr": 0.005, "n": 100000, "dist": {"dist_type": "exponpow", "dist_args": {"b": 2}, "samples": 10000}},'
    '{"p_ctr": 0.010, "n":  50000, "dist": {"dist_type": "exponpow", "dist_args": {"b": 2}, "samples": 10000}},'
    '{"p_ctr": 0.005, "n": 100000, "dist": {"dist_type": "exponpow", "dist_args": {"b": 4}, "samples": 10000}},'
    '{"p_ctr": 0.010, "n":  20000, "dist": {"dist_type": "exponpow", "dist_args": {"b": 4}, "samples": 10000}},'
    '{"p_ctr": 0.005, "n": 200000, "dist": {"dist_type": "exponpow", "dist_args": {"b": 6}, "samples": 10000}},'
    '{"p_ctr": 0.010, "n":  60000, "dist": {"dist_type": "exponpow", "dist_args": {"b": 6}, "samples": 10000}}'
    ']',
    help='Combinations of CTR probabilities, expected number of incoming '
         'requests, and auction winning bid distributions')
@click.option(
    '--save-path',
    type=str,
    default='data.json',
    help='Path to store the generated data')
def generate_data(auction_groups, save_path):
    """ Generate win-bid samples for different ad-auction groups """

    auction_groups = json.loads(auction_groups)

    logger.info(f'Generate data for the following auction groups:\n{auction_groups}')

    # p_ctr: CTR probabilities
    # n: expected number of incoming requests
    # winbid_samples: samples from the win-bid distribution
    data = [{'p_ctr': ac['p_ctr'],
             'n': ac['n'],
             'winbid_samples': generate_winbid_samples(**ac['dist'])}
            for ac in auction_groups]

    logger.info(f'Store data in {save_path}.')
    with open(save_path, "w") as f:
        json.dump(data, f)


@cli.command()
@click.option(
    '--data-path',
    type=str,
    default='data.json',
    help='Path where information for different auction groups is stored')
@click.option(
    '--n-click',
    type=float,
    default=100.,
    help='Desired number of user ad clicks')
def optimize_from_sample_date(data_path, n_click):
    """ Use empirical data for different auction types to construct an optimal
    bidding strategy """

    with open(data_path, "r") as f:
        data = json.load(f)

    logger.info(f'Loaded data for {len(data)} auction types.')

    # Click-through probabilities
    p_ctr = np.array([x['p_ctr'] for x in data])

    # Expected number of auctions per auction type
    n = np.array([x['n'] for x in data])

    # Fn that maps bids for different auction types to winning probabilities
    p_win_fn = winbid_prob_from_samples(
        samples_groups=[np.array(x['winbid_samples']) for x in data])

    # Expected number of clicks if we win every auction
    nc_max = (n * p_ctr).sum()

    assert n_click < nc_max, "The desired amount of clicks is lower than the " \
                             "expected number of clicks if we win every auction"

    # Initial value
    x0 = np.ones_like(p_ctr) * .001
    res = optimize(x0=x0, p_win_fn=p_win_fn, p_ctr=p_ctr, n=n, n_click=n_click)


@cli.command()
@click.option(
    '--auction-groups',
    type=str,
    default=
    '['
    '{"p_ctr": 0.005, "n": 100000, "dist": {"dist_type": "exponpow", "dist_args": {"b": 2}}},'
    '{"p_ctr": 0.010, "n":  50000, "dist": {"dist_type": "exponpow", "dist_args": {"b": 2}}},'
    '{"p_ctr": 0.005, "n": 100000, "dist": {"dist_type": "exponpow", "dist_args": {"b": 4}}},'
    '{"p_ctr": 0.010, "n":  20000, "dist": {"dist_type": "exponpow", "dist_args": {"b": 4}}},'
    '{"p_ctr": 0.005, "n": 200000, "dist": {"dist_type": "exponpow", "dist_args": {"b": 6}}},'
    '{"p_ctr": 0.010, "n":  60000, "dist": {"dist_type": "exponpow", "dist_args": {"b": 6}}}'
    ']',
    help='Combinations of CTR probabilities, expected number of incoming '
         'requests, and auction winning bid distributions')
@click.option(
    '--n-click',
    type=float,
    default=100.,
    help='Desired number of user ad clicks')
def optimize_from_analytical_dists(auction_groups, n_click):
    """ Use known distributions for different auction types to construct an optimal
    bidding strategy """

    data = json.loads(auction_groups)

    # Click-through probabilities
    p_ctr = np.array([x['p_ctr'] for x in data])

    # Expected number of auctions per auction type
    n = np.array([x['n'] for x in data])

    # Fn that maps bids for different auction types to winning probabilities
    p_win_fn = winbid_prob_analytical(dists=[x['dist'] for x in data])

    # Expected number of clicks if we win every auction
    nc_max = (n * p_ctr).sum()

    assert n_click < nc_max, "The desired amount of clicks is lower than the " \
                             "expected number of clicks if we win every auction"

    # Initial value
    x0 = np.ones_like(p_ctr) * .001
    res = optimize(x0=x0, p_win_fn=p_win_fn, p_ctr=p_ctr, n=n, n_click=n_click)


if __name__ == "__main__":
    """
    PYTHONPATH=src python src/main.py --help
    
    PYTHONPATH=$(pwd) python src/main.py optimize \
        --model_wrapper=ModelPDF \
        --model_args='{"l2": 0.0001, "batch_normalization": false, "layer_sizes": [64, [64, 64], [64, 64], 32, 8]}' \
        --ds_args='{"max_files": 2}' \
        --callbacks_args='{"period": 100, "profile_batch": 0}' \
        --training_args='{"epochs": 3000}'
    """
    cli()
