import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = 'df_revenue.csv'
    revenue_dataFrame = pd.read_csv(path) #columns are 'alt_name_', 'q_first', 'ROP_first', 'Revenue_mean', 'Revenue_std', 'CI_min', 'CI_max'
    revenue_dataFrame.plot.scatter(x="q_first", y="ROP_first", c="CI_min", title='q vs rop on ci min', colormap='summer_r', s=50) #_r in color map make the color reverse, s is for size revenue_dataFrame["CI_min"] * 0.005

    plt.show()