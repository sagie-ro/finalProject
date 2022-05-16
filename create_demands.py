import pandas as pd
import numpy as np
import scipy.stats as st


def PosNormal(mean, sigma):
    x = np.random.normal(mean, sigma, 1)
    return (x if x >= 0 else PosNormal(mean, sigma))


def create_yearly_demand(mean, sigma, distribution):
    # get the numbers
    demandArr = []
    for i in range(0, 365):
        demandArr.append(np.asscalar(distribution(mean, sigma).astype(int)))
    return demandArr
