import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import glob

sns.set(rc={'figure.figsize':(12,9)}, font_scale=2, style='darkgrid')
colors = sns.color_palette('colorblind', 4)
sns.set_palette(colors)

def fig():
    fig = 1
    while True:
        yield fig
        fig += 1
fig_gen = fig()

def moving_average(interval, window_size):
    if window_size == 1:
        return interval
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def plot_figure(figsize=(12, 9), x_label='', y_label='', title=''):
    ax = plt.subplot()

    plt.grid(axis='y')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    prs.add_argument("-f", dest="file", nargs='+', required=True, help="The csv file to plot.\n")
    prs.add_argument("-label", dest="label", nargs='+', required=False, help="Figure labels.\n")
    prs.add_argument("-out", dest="out", required=False, default='', help="The .pdf filename in which the figure will be saved.\n")
    prs.add_argument("-w", dest="window", required=False, default=5, type=int, help="The moving average window.\n")
    args = prs.parse_args()
    if args.label:
        labels = args.label
    else:
        labels = ['' for _ in range(len(args.file))]

    plot_figure(x_label='Trip departure time (s)', y_label='Trip duration(s)')

    for filename in args.file:
        main_df = pd.DataFrame()
        for file in glob.glob(filename+'*'):
            df = pd.read_csv(file, sep=';')
            if main_df.empty:
                main_df = df
            else:
                main_df = pd.concat((main_df, df))
        x = main_df['tripinfo_depart']
        y = main_df['tripinfo_duration']
        #x = main_df.groupby('tripinfo_depart').mean['tripinfo_depart']
        #y = main_df.groupby('tripinfo_depart').mean['tripinfo_duration']

        plt.plot(x,y)
        labels.pop(0)
  

    if args.label is not None:
        plt.legend()

    if args.out != '':
        plt.savefig(args.out+'.pdf', bbox_inches="tight")
    plt.show()
