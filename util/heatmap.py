import os
import time
import rospy
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def make_dir(dir):
    if(not os.path.exists(dir)):
        os.makedirs(dir)

def plot_heat_map(data,names,num,acc,an,line,ax):                   #### Number Float                                                ##### font size
    sns.heatmap(data, cmap='YlGnBu', annot=an, fmt='.2f', cbar_kws={'label': 'Color bar label'}, linewidths=.2, annot_kws={'size': 5}, xticklabels=list(np.arange(-num,num+acc,acc)), yticklabels=list(np.arange(num,-num-acc,-acc)))
    # Set the axis labels and title
    plt.xlabel('Yaw')
    plt.ylabel('Pitch')
    plt.axis(ax)
    plt.title(f'{names}_Error')

def main(cali,data):
    plot_heat_map(data)

if __name__ == "__main__":

    t1 = time.time()
    main()
    plt.show()
