import os, sys
from glob import glob
sys.path.insert(0, os.path.abspath('..'))
from whakaari import ForecastModel, load_dataframe, makedir

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

### --- ORIGINAL output vs probability of eruption plots Refer to https://dempseyresearchgroup.slack.com/archives/C01E27W4A5R/p1607477796071700
# Used for path to ORIGINAL version of sigmoid calibration
path_to_predictions = "/Users/teban/Documents/ADMIN/2020-21 Summer RA/PROGS/week3 plot progress/delta plots v2/predictions/"

# Custom plot metrics
colour_dict = {
    '4_month' : 'tomato',
    '3_month' : 'teal',
    '2_month' : 'goldenrod',
    '1_month' : 'navy',
}

# Marker_dict for eruption_num
marker_dict = {
    '0' : 'o',
    '1' : 'v',
    '2' : 'X',
    '3' : '*',
    '4' : 'D',
}


def stacked_by_eruption_plot(source=None, eruption_num=4, save=False, save_path=None, show=False):
    """ Produces stacked plot for eruption_num

        Parameters:
        -----------
        source : str
            path to folder of predictions
            NOTE: the predictions should end with [eruption_num]
        eruption_num : str, int
            eruption to extract
        save : bool
            whether or not to save the plots
        show : bool
            whether or not to show the plots

    """
    # get path to csvs
    if source is None:
        csvs = glob(f"{path_to_predictions}*__{eruption_num}.csv")
    else:
        csvs = glob(f"{source}*__{eruption_num}.csv")
    csvs.sort()
    print(csvs)

    # plot 'em together
    # ==== plot of calibrated probabilities vs thresholds ====
    f, ax = plt.subplots(1, 1, figsize=(18, 12))

    plt.axvline(0.8, color='pink', linewidth=5, zorder=1, label="Threshold")

    for csv in csvs:
        calib_df = pd.read_csv(csv)
        name = f"{csv.split('calibration_vals__')[-1].split('.')[0]}"
        test_window = name.split('__test_window_')[-1][:7]
        plt.scatter(calib_df['consensus'], calib_df['calibrated_prob'], s=60, zorder=2, alpha = 0.6,
            label=name.replace('__',', ').replace('_', ' '), color=colour_dict[test_window], marker=marker_dict[eruption_num])
            # label=name.replace('__',', ').replace('_', ' '), color=colour_dict[eruption_num], marker=marker_dict[test_window])

    plt.xlabel("MockForecastModel output", fontsize=25)
    plt.ylabel("Calibrated output Probability of eruption", fontsize=25)
    plt.title(f"Output vs probability of eruption test_window STACKED te = {eruption_num}", fontsize=32)
    for t in ax.get_xticklabels() + ax.get_yticklabels():  # increase of x and y tick labels
        t.set_fontsize(20.)

    ax.legend(fontsize=12)
    if save:
        if save_path is None:
            save_path = f"{path_to_predictions}{os.sep}plots"
            makedir(save_path)
        plt.savefig(f"{save_path}{os.sep}glorious_stacked_by_eruption_plot__{eruption_num}.png", format='png', dpi=300)
    if show: plt.show()
    plt.close()


def stacked_by_test_window_plot(source=None, test_window=4, save=False, save_path=None, show=False):
    """ Produces stacked plot for eruption_num

        Parameters:
        -----------
        source : str
            path to folder of predictions
            NOTE: the predictions should contain "__test_window_[test_window]"
        test_window : str, int
            test_window to extract
        save : bool
            whether or not to save the plots
        show : bool
            whether or not to show the plots

    """
    # get path to csvs
    if source is None:
        csvs = glob(f"{path_to_predictions}*__test_window_{test_window}*.csv")
    else:
        csvs = glob(f"{source}*__test_window_{test_window}*.csv")
    csvs.sort()
    print(csvs)

    # plot 'em together
    # ==== plot of calibrated probabilities vs thresholds ====
    f, ax = plt.subplots(1, 1, figsize=(18, 12))

    plt.axvline(0.8, color='pink', linewidth=5, zorder=1, label="Threshold")

    for csv in csvs:
        calib_df = pd.read_csv(csv)
        name = f"{csv.split('calibration_vals__')[-1].split('.')[0]}"
        eruption_num = name[-1]
        plt.scatter(calib_df['consensus'], calib_df['calibrated_prob'], s=60, zorder=2, alpha = 0.6,
            label=name.replace('__',', ').replace('_', ' '), color=colour_dict[test_window], marker=marker_dict[eruption_num])

    plt.xlabel("MockForecastModel output", fontsize=25)
    plt.ylabel("Calibrated output Probability of eruption", fontsize=25)
    plt.title(f"Output vs probability of eruption test_window={test_window}", fontsize=32)
    for t in ax.get_xticklabels() + ax.get_yticklabels():  # increase of x and y tick labels
        t.set_fontsize(20.)

    ax.legend(fontsize=12)
    if save:
        if save_path is None:
            save_path = f"{path_to_predictions}{os.sep}plots"
            makedir(save_path)
        plt.savefig(f"{save_path}{os.sep}glorious_stacked_by_test_window_plot__{test_window}.png", format='png', dpi=300)
    if show: plt.show()
    plt.close()


def stacked_by_all_plot(source=None, save=False, save_path=None, show=False):
    """ Produces stacked plot for eruption_num

        Parameters:
        -----------
        source : str
            path to folder of predictions
            NOTE: the predictions should contain "__test_window_[test_window]"
        test_window : str, int
            test_window to extract
        save : bool
            whether or not to save the plots
        show : bool
            whether or not to show the plots

    """
    # get path to csvs
    if source is None:
        csvs = glob(f"{path_to_predictions}*.csv")
    else:
        csvs = glob(f"{source}*.csv")
    csvs.sort()
    print(csvs)

    # plot 'em together
    # ==== plot of calibrated probabilities vs thresholds ====
    f, ax = plt.subplots(1, 1, figsize=(18, 12))

    plt.axvline(0.8, color='pink', linewidth=5, zorder=1, label="Threshold")

    for csv in csvs:
        calib_df = pd.read_csv(csv)
        name = f"{csv.split('calibration_vals__')[-1].split('.')[0]}"
        eruption_num = name[-1]
        if eruption_num == '3': continue
        test_window = name.split('__test_window_')[-1][:7]
        plt.scatter(calib_df['consensus'], calib_df['calibrated_prob'], s=60, zorder=2, alpha = 0.6,
            label=name.replace('__',', ').replace('_', ' '), color=colour_dict[test_window], marker=marker_dict[eruption_num])

    plt.xlabel("MockForecastModel output", fontsize=25)
    plt.ylabel("Calibrated output Probability of eruption", fontsize=25)
    plt.title(f"Output vs probability of eruption STACK ALL", fontsize=32)
    for t in ax.get_xticklabels() + ax.get_yticklabels():  # increase of x and y tick labels
        t.set_fontsize(20.)

    ax.legend(fontsize=12)
    if save:
        if save_path is None:
            save_path = f"{path_to_predictions}{os.sep}plots"
            makedir(save_path)
        plt.savefig(f"{save_path}{os.sep}glorious_stacked_by_all_plot.png", format='png', dpi=300)
    if show: plt.show()
    plt.close()
### ---------------------------------------------------------------


if __name__ == '__main__':
    ### --- output vs probability of eruption plots Refer to https://dempseyresearchgroup.slack.com/archives/C01E27W4A5R/p1607477796071700
    # stacked_by_eruption_plot()
    # stacked_by_test_window_plot()
    stacked_by_all_plot(save=True)

    ### --- timeline plots
    fig1()

