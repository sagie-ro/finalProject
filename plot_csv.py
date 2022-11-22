import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def scatter_plot(df):
    ax = df.plot.scatter(x="q_first", y="ROP_first", c="CI_min", title='q vs rop on ci min',colormap='RdYlGn',s=120)  # _r in color map make the color reverse, s is for size revenue_dataFrame["CI_min"] * 0.005
    ax.ticklabel_format(useOffset=False)
    plt.show()

def hexplot(df):
    ax = df.plot.hexbin(x="q_first", y="ROP_first", C="CI_min", gridsize=20, cmap="RdYlGn",
                                       title='q vs rop on ci min')
    ax.ticklabel_format(useOffset=False)
    plt.show()

if __name__ == '__main__':
    path = 'df_revenue.csv'
    revenue_dataFrame = pd.read_csv(path) #columns are 'alt_name_', 'q_first', 'ROP_first', 'Revenue_mean', 'Revenue_std', 'CI_min', 'CI_max'

    #scatter_plot(revenue_dataFrame)
    #hexplot(revenue_dataFrame)

    revenue_dataFrame.drop_duplicates(['q_first', 'ROP_first'], inplace=True)
    pivot = revenue_dataFrame.pivot(columns='q_first', index='ROP_first', values='Revenue_mean')
    ax = sns.heatmap(pivot, annot=True, cmap="RdYlGn", fmt='g')
    ax.set_title('Q vs Rop, Heatmap of Revenue_mean')

    plt.show()