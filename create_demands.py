import pandas as pd
import numpy as np
import scipy.stats as st


def PosNormal(mean, sigma):
    x = np.random.normal(mean, sigma, 1)
    return (x if x >= 0 else PosNormal(mean, sigma))

def Unif(min, max):
    x = np.random.uniform(min, max, 1)
    return (x if x >= 0 else Unif(min, max))


def create_yearly_demand(paramdict:dict, distribution:str):
    """
    function to create array with 365 values
    :param mean:
    :param sigma:
    :param distribution: function to create the value in the array
    :return: array of demands
    """
    # get the numbers
    demandArr = []
    if distribution =='normal':
        for i in range(0, 365):
            demandArr.append(np.asscalar(PosNormal(paramdict['mean'], paramdict['sigma']).astype(int)))
    elif distribution =='uniform':
        for i in range(0, 365):
            demandArr.append(np.asscalar(Unif(paramdict['min'], paramdict['max']).astype(int)))

    return demandArr
