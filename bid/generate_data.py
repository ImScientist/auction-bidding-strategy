import scipy.stats as sp


def generate_datasets(distribution_type: str, dist_params_list, sizes):
    """

    :param distribution_type: distribution from scipy.stats (for example 'expon')
    :param dist_params_list:
    :param sizes: list of numbers of generated samples from every distribution
    :return:
    """
    datasets = []

    for size, dist_params in zip(sizes, dist_params_list):

        ps = getattr(sp, distribution_type)(**dist_params)
        dataset = ps.rvs(size=size)
        datasets.append(dataset)

    return datasets
