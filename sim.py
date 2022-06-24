import pandas as pd
import numpy as np
import scipy.stats as st
import create_demands as cd
import eoq
import math


def norm_calc_rop(alpha, lt, sigma, mean):
    z = st.norm.ppf(alpha)
    b = z * (lt ** 0.5) * sigma
    rop = mean * lt + b
    return z, b, rop


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


def create_sim(mean, sigma, lt, k, c, interest, alpha, p, h=0, dist_func="normal", q_list=[0], file_name=None):
    # calc rop
    if dist_func == "normal":
        z, b, rop = norm_calc_rop(alpha, lt, sigma, mean)
    if h == 0:
        h = (interest * c)

    demand_arr = cd.create_yearly_demand(mean, sigma, cd.PosNormal)
    q_list = create_heuristic_q(demand_arr, q_list, k, mean, h, lt, rop)
    q_list = [math.ceil(item) for item in q_list]

    sim_df = []
    summary_list = []
    cumsum = []

    for Q in q_list:
        sm_d, sl, cc = sim_runner(demand_arr, Q, rop, lt, h, k, c, p, b)
        sim_df.append(sm_d)
        summary_list.append(sl)
        cumsum.append(cc)

    params = pd.DataFrame(data=[[mean, sigma, lt, k, c, interest, alpha, p, h, dist_func]],
                          columns=['mean', 'sigma', 'lt', 'k', 'c', 'interest', 'alpha', 'p', 'h', 'dist_func'])

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


def sim_runner(demand_arr, q, rop, lt, h, k, c, p, b):
    sim_df = pd.DataFrame(
        columns=['Demand', 'Inventory start day', 'Inventory end day', 'days until new supply arrives',
                 'inventory cost', 'order cost', 'item cost', 'total daily cost', 'total units cost', 'daily profit',
                 'total daily income', 'shortage'])

    for i in range(0, len(demand_arr)):
        # check inventory
        if i == 0:
            initial_storage = q
            # -1 indicates no new order
            days_to_new_supply = -1
        else:
            initial_storage = end_storage
        end_storage = max(initial_storage - demand_arr[i], 0)
        shortage = -(min(initial_storage - demand_arr[i], 0))
        sold_units = min(demand_arr[i], initial_storage)

        # check if arrived new items
        if days_to_new_supply == 1:
            end_storage = end_storage + q
            days_to_new_supply = -1
        elif days_to_new_supply > 0:
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
        day = [demand_arr[i], initial_storage, end_storage, days_to_new_supply, inventory_cost, order_cost, item_cost,
               total_daily_cost, sold_units, sold_units * p, sold_units * p - total_daily_cost, shortage]
        day = pd.DataFrame(np.array([day]), columns=['Demand', 'Inventory start day', 'Inventory end day',
                                                     'days until new supply arrives', 'inventory cost', 'order cost',
                                                     'item cost', 'total daily cost', 'total units cost',
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


create_sim(mean=1000, sigma=120, lt=20, k=1000, c=150, interest=0.1, h=150 * 0.1, alpha=0.95, p=200, dist_func="normal",
           q_list=[0, 1, 2, 3, 4, 5, 6, 7])
