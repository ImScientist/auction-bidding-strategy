import argparse
import numpy as np
import pandas as pd
from bid.generate_data import generate_datasets
from bid.get_pdfs import get_ps_pw_from_data, get_ps_pw_analytical
from bid.optimization import optimize


def get_summary(res, dist_params_list, pc, N):
    df = pd.DataFrame(index=pd.Index(np.arange(len(N))))
    df['x'] = res.x
    df['dist_params'] = dist_params_list
    df['dist_params'] = df['dist_params'].map(str)
    df['N_group'] = N
    df['pc'] = pc
    return df


if __name__ == "__main__":
    """ 
    Example:
        python main.py --distribution_type exponpow
    
    I: number of types of auction winning bid distributions
    J: number of different user click through probabilities
    Nc: desired number of user ad clicks
    
    N: N[j * I + i] = N_ij number cases where 
        auctions are of type i and 
        click through probabilities are of type j
        
    pc: pc[j * I + i] = pc_j 
        pc_j - user click through probability
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--distribution_type', type=str, dest='distribution_type', default='exponpow',
        help='Type of the winning bid probability distribution functions')

    args = parser.parse_args()
    distribution_type = args.distribution_type

    I = 3
    J = 2
    Nc = 100
    N_scale = 100000

    np.random.seed(11)
    N = np.random.randint(low=1, high=10, size=(I * J,)) * N_scale

    # user click through probabilities
    #
    pc = np.arange(1, J + 1, 1) * 0.005
    pc = np.repeat(pc, I)

    print('\n\nCheck if N*pc >> Nc\n')
    print('N*pc:      \t {0}'.format(int((N * pc).sum())))
    print('Nc:        \t {0}'.format(Nc))
    print('N*pc >> Nc \t {0}'.format((N * pc).sum() >= 10 * Nc))
    print('\n\n')

    # auction winning bid distributions
    #

    if distribution_type == 'exponpow':
        bs = np.arange(0, I) + 2
        dist_params_list = [dict(b=b) for b in bs]
    else:
        distribution_type = 'expon'
        alphas = np.arange(1, I + 1) * 0.1
        dist_params_list = [dict(scale=1. / a) for a in alphas]

    sizes = [sum([N[j * I + i] for j in range(J)]) for i in range(I)]

    # initial value
    #
    x0 = np.ones(shape=(I * J,)) * 0.001

    for case in ['fct analytical', 'fct inferred from data']:
        if case == 'fct inferred from data':
            datasets = generate_datasets(distribution_type, dist_params_list, sizes)
            ps, _, pw = get_ps_pw_from_data(datasets, J)
        else:
            ps, _, pw = get_ps_pw_analytical(J, distribution_type, dist_params_list)

        res = optimize(x0, pw, N, pc, Nc)
        df = get_summary(res, np.tile(dist_params_list, J), N)
        print('\n\nCase: {0}'.format(case))
        print(df)
