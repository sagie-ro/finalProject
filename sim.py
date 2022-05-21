import pandas as pd
import numpy as np
import scipy.stats as st
import create_demands as cd
import matplotlib.pyplot as plt
import designer as design
import xlsxwriter
import eoq

def norm_calc_rop(alpha, LT, sigma, mean):
    z = st.norm.ppf(alpha)
    b = z * (LT ** 0.5) * sigma
    rop = mean * LT + b
    return(z, b, rop)

def create_huristic_q(demand_list, huristic_list, k, mean, h, LT = 0, rop=0):
    Q_list = []
    Q_list.append(eoq.calc_eoq_1(k, mean, h))

    if 1 in huristic_list:
        Q_list.append((sum(demand_list)/len(demand_list))*LT)

    if 2 in huristic_list:
        Q_list.append(max(q for q in demand_list if q != 0) * LT)

    if 3 in huristic_list:
        Q_list.append(min(q for q in demand_list if q != 0) * LT)

    if 4 in huristic_list:
        Q_list.append(sum(demand_list))

    if 5 in huristic_list:
        Q_list.append(rop * LT)

    if 6 in huristic_list:
        Q_list.append(max(Q_list[0] + (sum(demand_list)/len(demand_list)),0))

    if 7 in huristic_list:
        Q_list.append(max(Q_list[0] - (sum(demand_list)/len(demand_list)),0))

    return Q_list

def create_sim(mean, sigma, LT, k, c, interest, alpha, p, distFunc = "normal", q_list = [0]):
    # calc rop
    if distFunc == "normal":
        z, b, rop = norm_calc_rop(alpha, LT, sigma, mean)
    h = interest * c

    demandArr = cd.create_yearly_demand(mean, sigma, cd.PosNormal)
    Q_list = create_huristic_q(demandArr, q_list, k, mean, h, LT, rop)
    Q_list = [int(item) for item in Q_list]

    simDf = []
    summary_list =[]
    cumsum = []

    for Q in Q_list:
        smD, sl, cc = sim_runner(demandArr, Q, rop, LT, h, k, c, p, b)
        simDf.append(smD)
        summary_list.append(sl)
        cumsum.append(cc)

    save_to_excel(simDf, summary_list, cumsum)

def save_to_excel(simDf, summary_list, cumsum):
    # save the result to exel
    writer = pd.ExcelWriter('res.xlsx', engine='xlsxwriter', options={'in_memory': True})

    for index in range(len(simDf)):
        simDf[index].to_excel(writer, sheet_name='Q' + str(index) + ' - daily')
        summary_list[index].to_excel(writer, sheet_name='Q' + str(index) + ' - daily', index=False,
                          startcol=len(simDf[index].columns) + 2)
        cumsum[index].to_excel(writer, sheet_name='Q' + str(index) + ' - cumulative sum')

    writer.save()

def sim_runner(demandArr, Q, rop, LT, h, k, c, p, b):
    simDf = pd.DataFrame(columns=['Demand', 'Inventory start day', 'Inventory end day', 'days untill new supply arrives',
                 'inventory cost', 'order cost', 'item cost', 'total daily cost', 'total units cost', 'daily profit',
                 'total daily income'])

    for i in range(0, len(demandArr)):
        # check inventory
        if i == 0:
            IS = Q
            # -1 indicates no new order
            daysToNewSupply = -1
        else:
            IS = ES
        ES = max(IS - demandArr[i], 0)
        sold_units = min(demandArr[i], IS)

        # check if arrived new items
        if daysToNewSupply == 1:
            ES = ES + Q
            daysToNewSupply = -1
        elif daysToNewSupply > 0:
            daysToNewSupply = daysToNewSupply - 1

        # check to create invite
        if ES <= rop and daysToNewSupply == -1:
            daysToNewSupply = LT

        # calc the cost
        inventory_cost = ES * h
        if (daysToNewSupply == LT):
            order_cost = k
            item_cost = Q * c
        else:
            order_cost = 0
            item_cost = 0

        total_daily_cost = inventory_cost + order_cost + item_cost
        day = [demandArr[i], IS, ES, daysToNewSupply, inventory_cost, order_cost, item_cost, total_daily_cost,
               sold_units, sold_units * p, sold_units * p - total_daily_cost]
        day = pd.DataFrame(np.array([day]), columns=['Demand', 'Inventory start day', 'Inventory end day',
                                                     'days untill new supply arrives', 'inventory cost', 'order cost',
                                                     'item cost', 'total daily cost', 'total units cost',
                                                     'daily profit', 'total daily income'])
        simDf = simDf.append(day)

    simDf = simDf.reset_index(drop=True)
    simDf.index += 1

    # create summary as: Q, b, Rop, Y(Q), G(Q), TC(Q)
    summary_list = [Q, b, rop, len(simDf[simDf['days untill new supply arrives']==LT]), simDf['inventory cost'].sum() + simDf['order cost'].sum(), simDf['total daily cost'].sum(), simDf['total daily income'].sum()]
    summary_list = pd.DataFrame(np.array([summary_list]), columns=['Q', 'b', 'ROP', 'how many orders', 'Y(Q)', 'G(Q)', 'TC(Q)'])

    #create comsum
    cumsum = pd.DataFrame()
    cumsum['GQ'] = simDf['total daily cost'].cumsum()
    cumsum['YQ'] = cumsum['GQ'] - simDf['item cost'].cumsum()
    cumsum['Income'] = simDf['daily profit'].cumsum()
    cumsum['TCQ'] = simDf['total daily income'].cumsum()
    temp = cumsum['YQ'].copy()
    cumsum['YQ'] = cumsum['GQ']
    cumsum['GQ'] = temp
    cumsum = cumsum.rename(columns={'GQ':'YQ', 'YQ':'GQ'})

    return simDf, summary_list, cumsum

create_sim(mean = 10, sigma = 10, LT = 3
           , k = 100, c = 100, interest = 0.1,
           alpha = 0.5, p = 150, distFunc = "normal", q_list = [0,1,2,3,4,5,6,7])

# plots: day vs Inventory start day, day vs Y(Q), day vs G(Q), day vs TC(Q)
#plt.ticklabel_format(style='plain')
#simDf.plot(y='Inventory start day', use_index=True)
#TCQ.plot.line(use_index=True, title='TCQ vs days', grid=True)
#TCQ.plot.line(use_index=True, title='TCQ vs days', grid=True)
#TCQ.plot.line(use_index=True, title='TCQ vs days', grid=True)
#TCQ.plot.line(use_index=True, title='TCQ vs days', grid=True)
#plt.show()

