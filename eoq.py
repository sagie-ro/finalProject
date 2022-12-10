import scipy.stats as st
import math


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


def create_heuristic_q_rop(alpha, lt, sigma, baseMean, h, k, n=2):
    """
        Return dict of n heuristics (q,rop,b).
        :param alpha: service level
        :param lt: lead-time
        :param sigma: standard-deviation
        :param baseMean: mean of Demand
        :param h: inventory cost per unit per year
        :param k: Order cost
        :param n: number of heuristics (int)
        :return: heuristics (dict)
    """
    # init
    heuristics = {}
    count = 2
    increment_q = 1  # as default
    increment_rop = 1  # as default
    increment_ratio = 0.1

    # normal
    z, b, rop = norm_calc_rop(alpha, lt, sigma, baseMean)
    min_q = lt * baseMean
    q = calc_eoq_1(k, baseMean, h, min_q)
    heuristics[f'alt{1}'] = {'q': int(q), 'rop': int(rop), 'b': int(b)}

    increment_q = math.ceil(q * increment_ratio) # relative increment
    increment_rop = math.ceil(rop * increment_ratio)  # relative increment
    print(f"Simulation increment for q is set to: {increment_q}")
    print(f"Simulation increment for rop is set to: {increment_rop}")

    q = q - (increment_q * (int(math.sqrt(n))/2))

    for i in range(1 , n):
        z, b, rop = norm_calc_rop(alpha, lt, sigma, baseMean)
        rop -= increment_rop * ( math.sqrt(n)/2 )
        for j in range(1,int(math.sqrt(n))+1):
            heuristics[f'alt{count}'] = {'q': int(q), 'rop': int(rop), 'b': int(b)}
            rop+=increment_rop
            count+=1
            if count>n: break
        heuristics[f'alt{count}'] = {'q': int(q), 'rop': int(rop), 'b': int(b)}
        q += increment_q
        count +=1
        if count > n: break
    return heuristics

#print (create_heuristic_q_rop(0.95, lt=1, sigma=23, baseMean=109.58, h=1, k=5000, n=8))

'''

    # Uniform
    z, b, rop = unif_calc_rop(alpha, lt, min=baseMean - (baseMean / 2), max=baseMean + (baseMean / 2))
    min_q = lt * baseMean
    q = calc_eoq_1(k, baseMean, h, min_q)
    heuristics[f'alt{2}'] = {'q': int(q), 'rop': int(rop), 'b': int(b)}

    increment = 1
    posmean = baseMean  # add incerements
    negmean = baseMean  # substract increments
    posQ = q
    negQ = q
    posROP = rop
    negROP = rop
    for i in range(3, n+1):
        if i % 4 == 1:
            posQ += increment
            heuristics[f'alt{i}'] = {'q': int(posQ), 'rop': int(posROP), 'b': int(b)}

        if i % 4 == 2:
            posROP += increment
            heuristics[f'alt{i }'] = {'q': int(posQ), 'rop': int(posROP), 'b': int(b)}

        if i % 4 == 3:
            negQ -= increment
            heuristics[f'alt{i}'] = {'q': int(negQ), 'rop': int(negROP), 'b': int(b)}

        if i % 4 == 0:
            negROP -= increment
            heuristics[f'alt{i}'] = {'q': int(negQ), 'rop': int(negROP), 'b': int(b)}
'''
'''    
for i in range(3, n, 2):
        if i % 4 == 3:
            posmean += increment
            # normal
            z, b, rop = norm_calc_rop(alpha, lt, sigma, posmean)
            min_q = lt * posmean
            q = calc_eoq_1(k, posmean, h, min_q)

            heuristics[f'alt{i}'] = {'q': int(q), 'rop': int(rop), 'b': int(b)}

            # Uniform
            z, b, rop = unif_calc_rop(alpha, lt, min=posmean - (posmean / 2), max=posmean + (posmean / 2))
            min_q = lt * posmean
            q = calc_eoq_1(k, posmean, h, min_q)

            heuristics[f'alt{i + 1}'] = {'q': int(q), 'rop': int(rop), 'b': int(b)}

        if i % 4 == 1:
            negmean -= increment
            # normal
            z, b, rop = norm_calc_rop(alpha, lt, sigma, negmean)
            min_q = lt * negmean
            q = calc_eoq_1(k, negmean, h, min_q)

            heuristics[f'alt{i}'] = {'q': int(q), 'rop': int(rop), 'b': int(b)}

            # Uniform
            z, b, rop = unif_calc_rop(alpha, lt, min=negmean - (negmean / 2), max=negmean + (negmean / 2))
            min_q = lt * negmean
            q = calc_eoq_1(k, negmean, h, min_q)

            heuristics[f'alt{i + 1}'] = {'q': int(q), 'rop': int(rop), 'b': int(b)}
'''

