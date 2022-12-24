import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d

def scatter_plot(df,column_name):
    ax = df.plot.scatter(x="q_first", y="ROP_first", c=column_name, title='q vs rop on Revenue mean',colormap='RdYlGn',s=120)  # _r in color map make the color reverse, s is for size revenue_dataFrame["CI_min"] * 0.005
    ax.ticklabel_format(useOffset=False)
    plt.show()

def hexplot(df,column_name):
    ax = df.plot.hexbin(x="q_first", y="ROP_first", C=column_name, gridsize=20, cmap="RdYlGn",
                                       title='q vs rop on Revenue mean')
    ax.ticklabel_format(useOffset=False)
    plt.show()

def heatmap(df,column_name):
    df.drop_duplicates(['q_first', 'ROP_first'], inplace=True)
    pivot = df.pivot(columns='q_first', index='ROP_first', values=column_name)
    ax = sns.heatmap(pivot, annot=False, cmap="RdYlGn", fmt='g')
    ax.set_title(f'Q vs Rop, Heatmap of {column_name}')
    plt.show()

def plot3d(df,column_name):
    pivot = df[["q_first", "ROP_first", column_name]].drop_duplicates()
    pivot = pivot.pivot(index="q_first", columns="ROP_first", values=column_name)
    Y = pivot.columns
    X = pivot.index
    Z = pivot.values
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('ROP')
    ax.set_ylabel('q')
    ax.set_zlabel(column_name);
    plt.show()

if __name__ == '__main__':
    #path = "sum_heatmap_csv.csv"
    path = "sum_heatmap_5000_135k.csv"
    column_name = "Revenue_mean"
    #column_name = "Revenue_mean_calculated"
    revenue_dataFrame = pd.read_csv(path) #columns are 'alt_name_', 'q_first', 'ROP_first', column_name, 'Revenue_std', 'CI_min', 'CI_max'

    scatter_plot(revenue_dataFrame,column_name)
    hexplot(revenue_dataFrame,column_name)
    heatmap(revenue_dataFrame,column_name)
    plot3d(revenue_dataFrame,column_name)

    #get shortage of alt 1
    #shortage = float(revenue_dataFrame.loc[revenue_dataFrame["alt_name_"] == "alt1", "Shortage Percent_mean"])
    #shortage_df_less_than_alt_1 = revenue_dataFrame[revenue_dataFrame["Shortage Percent_mean"] <= shortage]
    #shortage_df_less_than_alt_1.to_csv('shortage less than alt 1.csv',index=False)
    #scatter_plot(shortage_df_less_than_alt_1)
    #hexplot(shortage_df_less_than_alt_1)
    #heatmap(shortage_df_less_than_alt_1)
    #plot3d(shortage_df_less_than_alt_1)
