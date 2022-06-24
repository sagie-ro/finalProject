def calc_eoq_1(k, mean, h, min_Q):
    '''
    Calculate optimal q for given K, mean and h.

    :param k: Order cost
    :param mean: mean of Demand
    :param h: inventory cost per unit per year
    :param min_Q: the min q to order so lt * invite per year < 365
    :return: Optimal q (float)
    '''
    mean_yearly = mean*365
    Q_opt = (2*k*mean_yearly/h)**0.5
    return max(Q_opt, min_Q)
