import pandas as pd
import numpy as np
import scipy.stats as st
import create_demands as cd
import eoq
import math
from scipy.stats import t

# todo run the simulation on unif with fixed demand
# todo run the simultion according to simulation theory of n
# todo create Confidence interval with params, show if this is in the range
# todo show the match equation showing the income vs the simulation

def create_heuristic_q(demand_list, heuristic_list, k, mean, h, lt=0, rop=0):
    """
    Creat list of q by heuristics. first is optimal

    :param demand_list: list of demand [int[]]
    :param heuristic_list: list of heuristic ID [int[]]
    :param k: value of order cost [float]
    :param mean: mean of demand [float]
    :param h: inventory cost per unit per year [float]
    :param lt: LeadTime value [int]
    :param rop: Re-order point [int]
    :return: Q_list: list of q [float[]]
    """
    q_list = []
    min_q = lt * mean
    q_list.append(eoq.calc_eoq_1(k, mean, h, min_q))

    if 1 in heuristic_list:  # calculate average Demand and multiply with lt
        q_list.append((sum(demand_list) / len(demand_list)) * lt)

    if 2 in heuristic_list:  # calculate Max Demand and multiply with lt
        q_list.append(max(q for q in demand_list if q != 0) * lt)

    if 3 in heuristic_list:  # calculate Min Demand and multiply with lt
        q_list.append(min(q for q in demand_list if q != 0) * lt)

    if 4 in heuristic_list:  # calculate sum Demand
        q_list.append(sum(demand_list))

    if 5 in heuristic_list:  # calculate Rop multiply with lt
        q_list.append(rop * lt)

    if 6 in heuristic_list:  # add daily average demand to optimal q
        q_list.append(max(q_list[0] + (sum(demand_list) / len(demand_list)), 0))

    if 7 in heuristic_list:  # subtract daily average demand to optimal q
        q_list.append(max(q_list[0] - (sum(demand_list) / len(demand_list)), 0))

    return q_list

def create_sim( lt, k, c, interest, alpha, p, paramdict:dict, dist_func="normal", q_list=[0], file_name=None, for_loop_sim=False):
    """
    call save to excel with simulation params
    :param paramdict:
    :param for_loop_sim: bool
    :param lt:
    :param k:
    :param c:
    :param interest:
    :param alpha:
    :param p:
    :param h:
    :param dist_func:
    :param q_list:
    :param file_name:
    :return:
    """

    # calc rop

    h = interest * c

    if for_loop_sim == False:
        if dist_func == "normal":
            z, b, rop = eoq.norm_calc_rop(alpha, lt, paramdict["sigma"], paramdict["mean"])


        elif dist_func == "uniform":
            z, b, rop = eoq.unif_calc_rop(alpha, lt, paramdict["min"], paramdict["max"])
            paramdict["mean"] = (paramdict["max"] + paramdict["min"]) / 2
            paramdict["sigma"] = (((paramdict["max"] - paramdict["min"]) ** 2) / 12) ** 0.5

        print(q_list)
        create_sim_regular(paramdict, lt, k, c, interest, alpha, p, h, rop, b,dist_func, q_list=q_list, file_name=file_name)

    # run sim loop
    else:
        create_sim_loop(paramdict, dist_func, lt, k, c, p, h, alpha)

'''
def create_sim_production(lt, k, c, interest, alpha, p, paramdict: dict, dist_func="normal", q_list=[0], file_name=None,
                          for_loop_sim=False):
    """
    call save to excel with simulation params
    :param paramdict:
    :param for_loop_sim: bool
    :param lt:
    :param k:
    :param c:
    :param interest:
    :param alpha:
    :param p:
    :param h:
    :param dist_func:
    :param q_list:
    :param file_name:
    :return:
    """

    # calc rop

    h = interest * c

    if for_loop_sim == False:
        if dist_func == "normal":
            z, b, rop = eoq.norm_calc_rop(alpha, lt, paramdict["sigma"], paramdict["mean"])


        elif dist_func == "uniform":
            z, b, rop = eoq.unif_calc_rop(alpha, lt, paramdict["min"], paramdict["max"])
            paramdict["mean"] = (paramdict["max"] + paramdict["min"]) / 2
            paramdict["sigma"] = (((paramdict["max"] - paramdict["min"]) ** 2) / 12) ** 0.5

        print(q_list)
        create_sim_regular(paramdict, lt, k, c, interest, alpha, p, h, rop, b, dist_func, q_list=q_list,
                           file_name=file_name)

    # run sim loop
    else:
        create_sim_loop(paramdict, dist_func, lt, k, c, p, h, alpha)
'''

def run_sim_once_return_sl(lt, k, c, p, h, rop, b, demand_arr, q_to_order):
    # generated the demands
    sm_d, sl, cc = sim_runner(demand_arr, q_to_order, rop, lt, h, k, c, p, b)
    return sl

def create_sim_loop(paramdict:dict, dist_func:str, lt, k, c, p, h, alpha, demand_by_n = [], n0=25):
    # create the q and rop
    q_rop_dict = eoq.create_heuristic_q_rop(alpha, lt, paramdict['sigma'], paramdict['mean'], h, k, n=5)
    print(q_rop_dict)
    print('times to run the sim', n0)

    #create demand arr #if t mzovag #if welch
    demands_to_make = n0 # for t, if welch = len(q_rop_dict) * n0
    for i in range(demands_to_make - len(demand_by_n)):
        demand_by_n.append(cd.create_yearly_demand(paramdict, dist_func))

    # run simulation in a loop
    for alternitive in q_rop_dict:
        sim_alt = q_rop_dict[alternitive]
        rop_when_order = sim_alt['rop']
        q_to_order = sim_alt['q']
        b = sim_alt['b']
        for n in range(len(demand_by_n)):
            # demand_arr = cd.create_yearly_demand(paramdict, dist_func) for welch
            demand_arr = demand_by_n[n] #t-paired
            if n == 0:
                summary_list = run_sim_once_return_sl(lt, k, c, p, h, rop_when_order, b, demand_arr, q_to_order)
            else:
                summary_list = summary_list.append(run_sim_once_return_sl(lt, k, c, p, h, rop_when_order, b, demand_arr, q_to_order))
        summary_list.reset_index(drop=True, inplace=True)
        summary_list['alt_name'] = alternitive
        if alternitive == 'alt1':
            sim_summary_runner = summary_list
        else:
            sim_summary_runner = sim_summary_runner.append(summary_list)
    heat_map_eoq,new_n = create_heatmap_q_rop(sim_summary_runner, n0)

    # run it enough times
    if new_n > n0:
        create_sim_loop(paramdict, dist_func, lt, k, c, p, h, alpha, demand_by_n, new_n)
    else:
        print(heat_map_eoq)

def create_t_paired(heat_map_eoq):
    print(heat_map_eoq)

def create_sim_regular(paramdict, lt, k, c, interest, alpha, p, h, rop, b, dist_func="normal", q_list=[0], file_name=None):
    # generated the demands #todo fit it to model 1
    print(q_list)

    demand_arr = cd.create_yearly_demand(paramdict, dist_func)
    q_list = create_heuristic_q(demand_arr, q_list, k, paramdict["mean"], h, lt, rop)
    q_list = [math.ceil(item) for item in q_list]

    # DataFrames of the summary
    sim_df = []
    summary_list = []
    cumsum = []

    # generate q, the amount to order
    for Q in q_list:
        # the function to create the simulation
        sm_d, sl, cc = sim_runner(demand_arr, Q, rop, lt, h, k, c, p,b)  # sm_d is the simulation itself, sl is the params and summaries, cc is the cumsum
        sim_df.append(sm_d)
        summary_list.append(sl)
        cumsum.append(cc)

    # write down the parameters
    params = pd.DataFrame(data=[[paramdict["mean"], paramdict["sigma"], lt, k, c, interest, alpha, p, h, dist_func]],
                              columns=['mean', 'sigma', 'lt', 'k', 'c', 'interest', 'alpha', 'p', 'h', 'dist_func'])

    # save all to excel
    save_to_excel(sim_df, summary_list, cumsum, params, file_name)


def save_to_excel(sim_df, summary_list, cumsum, params, file_name=None):
    # save the result to exel
    if file_name is None:
        file_name = ''
        for col in params.columns:
            file_name += str(col)
            file_name += str(params.loc[0, col])
            file_name += '_'
        file_name += 'res.xlsx'
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter', options={'in_memory': True})

    merge = pd.concat(summary_list)
    merge = merge.reset_index(drop=True)
    merge.to_excel(writer, sheet_name='summary', index=False)

    for index in range(len(sim_df)):
        sim_df[index].to_excel(writer, sheet_name='q' + str(index) + ' - daily', index=False)
        params.to_excel(writer, sheet_name='q' + str(index) + ' - daily', index=False,
                        startcol=len(sim_df[index].columns) + 2)
        rop_q = summary_list[index][['q', 'ROP']]
        vals = summary_list[index].drop(columns=['q', 'ROP'])
        vals.to_excel(writer, sheet_name='q' + str(index) + ' - daily', index=False,
                      startcol=len(sim_df[index].columns) + 2, startrow=3)
        rop_q.to_excel(writer, sheet_name='q' + str(index) + ' - daily', index=False,
                       startcol=len(sim_df[index].columns) + 2, startrow=6)
        cumsum[index].to_excel(writer, sheet_name='q' + str(index) + ' - cumulative sum', index=False)

    writer.save()


def sim_runner(demand_arr, production_rate, q, rop, lt, h, k, c, p, b):
    sim_df = pd.DataFrame(
        columns=['Demand', 'Production', 'Inventory start day', 'Inventory end day', 'days until new supply arrives',
                 'inventory cost', 'order cost', 'item cost', 'total daily cost', 'total units sold', 'daily profit',
                 'total daily income', 'shortage'])

    # the simulation itself
    for i in range(0, len(demand_arr)):
        # check inventory
        if i == 0:
            # we start the simulation when we have q in the storage
            initial_storage = q
            # -1 indicates no new order
            days_to_new_supply = -1
            # produced in a day
            production = 0
        else:
            # if this is not the first day, we start a new day with storage from yesterday
            initial_storage = end_storage
        # first we sell all that we can
        end_storage = max(initial_storage - demand_arr[i], 0)
        # set what we couldn't sell as shortage
        shortage = -(min(initial_storage - demand_arr[i], 0))
        # how much we sold that day
        sold_units = min(demand_arr[i], initial_storage)

        # check if arrived new items
        if days_to_new_supply == 1:
            end_storage = end_storage + production
            production = 0
            days_to_new_supply = -1

        elif days_to_new_supply > 1:
            production = production_rate
            end_storage = end_storage + production
            days_to_new_supply = days_to_new_supply - 1

        # check to create invite
        if end_storage <= rop and days_to_new_supply == -1:
            days_to_new_supply = lt

        # calc the cost
        inventory_cost = end_storage * h / 365
        if days_to_new_supply == lt or i == 0:
            order_cost = k
            item_cost = q * c
        else:
            order_cost = 0
            item_cost = 0

        total_daily_cost = inventory_cost + order_cost + item_cost
        day = [demand_arr[i],production, initial_storage, end_storage, days_to_new_supply, inventory_cost, order_cost, item_cost,
               total_daily_cost, sold_units, sold_units * p, sold_units * p - total_daily_cost, shortage]
        day = pd.DataFrame(np.array([day]), columns=['Demand', 'Production', 'Inventory start day', 'Inventory end day',
                                                     'days until new supply arrives', 'inventory cost', 'order cost',
                                                     'item cost', 'total daily cost', 'total units sold',
                                                     'daily profit', 'total daily income', 'shortage'])
        sim_df = sim_df.append(day)

    sim_df = sim_df.reset_index(drop=True)
    sim_df.index += 1

    # create summary as: q, b, Rop, Y(q), G(q), Revenue
    summary_list = [q, b, rop, len(sim_df[sim_df['days until new supply arrives'] == lt]),
                    sim_df['inventory cost'].sum() + sim_df['order cost'].sum(), sim_df['total daily cost'].sum(),
                    sim_df['total daily income'].sum(),
                    sim_df['shortage'].sum() / sum(demand_arr) * 100
                    ]
    summary_list = pd.DataFrame(np.array([summary_list]),
                                columns=['q', 'b', 'ROP', 'how many orders', 'Y(q)', 'G(q)', 'Revenue',
                                         'Shortage Percent'])

    # create cumsum
    cumsum = pd.DataFrame()
    cumsum['GQ'] = sim_df['total daily cost'].cumsum()
    cumsum['YQ'] = cumsum['GQ'] - sim_df['item cost'].cumsum()
    cumsum['Income'] = sim_df['daily profit'].cumsum()
    cumsum['Revenue'] = sim_df['total daily income'].cumsum()
    temp = cumsum['YQ'].copy()
    cumsum['YQ'] = cumsum['GQ']
    cumsum['GQ'] = temp
    cumsum = cumsum.rename(columns={'GQ': 'YQ', 'YQ': 'GQ'})

    return sim_df, summary_list, cumsum


def create_heatmap_q_rop(summary_q_rop, n):
    # creating the heatmap table
    heatmap_q_rop = summary_q_rop[['q', 'ROP', 'Revenue', 'alt_name']]
    heatmap_q_rop = heatmap_q_rop.groupby(['alt_name'], as_index=False).agg(
        {'q': 'first', 'ROP': 'first', 'Revenue': ['mean', 'std']})
    heatmap_q_rop.columns = heatmap_q_rop.columns.to_flat_index()
    heatmap_q_rop.columns = ['_'.join(col) for col in heatmap_q_rop.columns.values]


    # calculating confidence interval, with 5 percent
    t_crit = np.abs(t.ppf((0.05) / 2, n)) #todo check it
    heatmap_q_rop['CI_min'] = heatmap_q_rop['Revenue_mean']-heatmap_q_rop['Revenue_std']*t_crit/np.sqrt(n + 1)
    heatmap_q_rop['CI_max'] = heatmap_q_rop['Revenue_mean']+heatmap_q_rop['Revenue_std']*t_crit/np.sqrt(n + 1)
    heatmap_q_rop = heatmap_q_rop.sort_values(by='CI_max', ascending=False)
    heatmap_q_rop['half_CI'] = (t_crit * heatmap_q_rop['Revenue_std'])/((n+1)**0.5)
    gama = 0.1
    gama_tag = gama / (1 + gama)
    heatmap_q_rop['precision'] = heatmap_q_rop['half_CI']/heatmap_q_rop['Revenue_mean']
    heatmap_q_rop['N_to_make'] = (n+1)*((heatmap_q_rop['precision']/gama_tag)**2)
    heatmap_q_rop['N_to_make'] = heatmap_q_rop['N_to_make'].apply(np.ceil)
    heatmap_q_rop['N_to_make_more'] = heatmap_q_rop['N_to_make'] - n
    heatmap_q_rop.loc[heatmap_q_rop['N_to_make_more'] < 0, 'N_to_make_more'] = 0
    pd.set_option('display.float_format', str)
    pd.options.display.float_format = '{:.3f}'.formatד
    pd.set_option('display.max_columns', None)
    n_more_to_make = int(max(heatmap_q_rop['N_to_make_more']))
    heatmap_q_rop = heatmap_q_rop[['alt_name_', 'q_first', 'ROP_first', 'Revenue_mean', 'Revenue_std', 'CI_min', 'CI_max']]
    return heatmap_q_rop, n_more_to_make

if __name__ == '__main__':
    # if unif mean is min and sigma is max
    # create_sim(mean=1000, sigma=120, lt=20, k=1000, c=150, interest=0.1, alpha=0.95, p=200, dist_func="normal",
    #      q_list=[0], for_loop_sim=25)
    paramdict = {
        "mean": 109.58,
        "sigma": 23,
        "max": 109.58,
        "min": 109.58,
        "manufacture_mean": -1
    }
    if paramdict['manufacture_mean'] == -1:
        create_sim(paramdict=paramdict, lt=1, k=5000, c=10, interest=0.1, alpha=0.95, p=12, dist_func="normal",
                   q_list=[0], for_loop_sim=True)
'''
    else:
        create_sim_production(paramdict=paramdict, lt=1, k=5000, c=10, interest=0.1, alpha=0.95, p=12,
                              dist_func="normal",
                              q_list=[0],
                              for_loop_sim=True)
'''
