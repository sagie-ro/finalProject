import scipy.stats as st


def norm_calc_rop(alpha, lt, sigma, mean):
    z = st.norm.ppf(alpha)
    b = z * (lt ** 0.5) * sigma
    rop = mean * lt + b
    return z, b, rop


def unif_calc_rop(alpha, lt, min, max):
    mean = (max + min) / 2
    sigma = (((max - min) ** 2) / 12) ** 0.5
    z = st.norm.ppf(alpha)
    b = z * (lt ** 0.5) * sigma
    rop = mean * lt + b
    return z, b, rop


def calc_eoq_1(k, mean, h, min_Q):
    """
    Calculate optimal q for given K, mean and h.

    :param k: Order cost
    :param mean: mean of Demand
    :param h: inventory cost per unit per year
    :param min_Q: the min q to order so lt * invite per year < 365
    :return: Optimal q (float)
    """
    mean_yearly = mean * 365
    Q_opt = (2 * k * mean_yearly / h) ** 0.5
    return max(Q_opt, min_Q)


def create_heuristic_q_rop(alpha, lt, sigma, mean, h, k, n=4):
    """
        Return dict of n heuristics (q,rop,b).
        :param alpha: service level
        :param lt: lead-time
        :param sigma: standard-deviation
        :param mean: mean of Demand
        :param h: inventory cost per unit per year
        :param k: Order cost
        :param n: number of heuristics (int)
        :return: heuristics (dict)
    """

    heuristics = {}
    name = 'alt'
    for i in range(1, n + 1, 2):
        lt, sigma, mean = lt * i, sigma * i, mean * i
        # normal
        z, b, rop = norm_calc_rop(alpha, lt, sigma, mean)
        min_q = lt * mean
        q = calc_eoq_1(k, mean, h, min_q)

        heuristics[f'alt{i}'] = {'q': int(q), 'rop': int(rop), 'b': int(b)}

        # Uniform
        z, b, rop = unif_calc_rop(alpha, lt, min=mean - (mean / 2), max=mean + (mean / 2))
        min_q = lt * mean
        q = calc_eoq_1(k, mean, h, min_q)

        heuristics[f'alt{i + 1}'] = {'q': int(q), 'rop': int(rop), 'b': int(b)}

    return heuristics


#print(create_heuristic_q_rop(0.95, 1, 23, 109.58, 1, 5000, n=2))


