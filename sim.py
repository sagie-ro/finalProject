import pandas as pd
import numpy as np
import scipy.stats as st

mean = 10
sigma = 10
Q = 100
LT = 3
k = 100
c = 100
ribit = 0.1
alpha = 0.5 #half the time there might be a hoser

def PosNormal(mean, sigma):
    x = np.random.normal(mean, sigma,1)
    return(x if x>=0 else PosNormal(mean,sigma))

# get the numbers
demandArr = []
for i in range(0,365):
    demandArr.append(np.asscalar(PosNormal(mean, sigma).astype(int)))

# calc rop
z = st.norm.ppf(alpha)
b = z*(LT**0.5)*sigma
rop = mean*LT + b
h = ribit*c

# sim
simDf = pd.DataFrame(columns = ['Demand', 'Inventory start day', 'Inventory end day', 'days untill new supply arrives', 'inventory cost', 'order cost', 'item cost', 'total daily cost'])
for i in range(0, len(demandArr)):
    #check inventory
    if i == 0:
        IS = Q
        #-1 indicates no new order
        daysToNewSupply = -1
    else:
        IS = max(ES - demandArr[i], 0)
    ES = max(IS - demandArr[i], 0)

    #check if arrived new items
    if daysToNewSupply == 1:
        ES = ES + Q
        daysToNewSupply = -1
    elif daysToNewSupply >0:
        daysToNewSupply = daysToNewSupply - 1

    # check to create invite
    if ES <= rop and daysToNewSupply == -1:
        daysToNewSupply = LT

    #calc the cost
    inventory_cost = ES * h
    if(daysToNewSupply == LT):
        order_cost = k
        item_cost = Q*c
    else:
        order_cost = 0
        item_cost = 0

    total_daily_cost = inventory_cost + order_cost + item_cost
    day = [demandArr[i], IS, ES, daysToNewSupply, inventory_cost, order_cost, item_cost, total_daily_cost]
    day = pd.DataFrame(np.array([day]), columns=['Demand', 'Inventory start day', 'Inventory end day', 'days untill new supply arrives', 'inventory cost', 'order cost', 'item cost', 'total daily cost'])
    simDf = simDf.append(day)

print(simDf)

'''
check huristic for cost:
q1 = order the minimum of the demand *LT
q2 = order the max demand *LT
q3 = order the mean demand *LT
q4 = order sum of demand
q5 = order ROP *LT
q6 = order Q* +mean demand
q7 = order Q* - mean demand
'''

