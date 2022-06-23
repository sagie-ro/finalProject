def calc_eoq_1(k, mean, h):
    '''
    Calculate optimal Q for given K, mean and h.

    :param k: Order cost
    :param mean: mean of Demand
    :param h: inventory cost per unit per year
    :return: Optimal q (float)
    '''
    return (2*k*mean/h)**0.5
