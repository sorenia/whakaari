import os, sys
from glob import glob
sys.path.insert(0, os.path.abspath('..'))
from whakaari import TremorData, ForecastModel, load_dataframe, makedir, timedelta, datetimeify

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.calibration import _sigmoid_calibration
from scipy.special import expit

### --- ORIGINAL output vs probability of eruption plots Refer to https://dempseyresearchgroup.slack.com/archives/C01E27W4A5R/p1607477796071700
# Used for path to ORIGINAL version of sigmoid calibration
path_to_predictions = "/Users/teban/Documents/ADMIN/2020-21 Summer RA/PROGS/week3 plot progress/delta plots v2/predictions/"

# Custom plot metrics
colour_dict = {
    '4_month' : 'tomato',
    '3_month' : 'teal',
    '2_month' : 'goldenrod',
    '1_month' : 'navy',
    '0' : 'tomato',
    '1' : 'teal',
    '2' : 'goldenrod',
    '3' : 'violet',
    '4' : 'navy',
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


### --- fig1: Visualisation of constructing model output timeline
def fig1():


    data_streams = ['rsam', 'mf', 'hf', 'dsar']
    fm = ForecastModel(ti='2011-01-01', tf='2020-01-01', window=2., overlap=0.75,
                       look_forward=2., data_streams=data_streams, root=f'calibration_forecast_model', savefile_type='pkl')
    f_load = f"{fm.rootdir}/calibration/{fm.root}__hires_test__TIMELINE__ncl_500.pkl"
    try:
        timeline = load_dataframe(f_load, index_col='time')
    except FileNotFoundError:
        print("file {f_load} not found, please run construct_hires_timeline() [calibration.py]")
        return
    none_timeline = load_dataframe(f"{fm.rootdir}/calibration/{fm.root}__hires_test__NONE__ncl_500.pkl", index_col='time')

    fig, ax = plt.subplots(figsize=[8,3])
    fig.tight_layout
    a_index = timeline.index[0]
    insertion_enums = [0,1,2,4]
    for eruption_num in insertion_enums:
        insert = timeline.loc[timeline.model==eruption_num]
        b_index = insert.index[0]
        # plot before
        none_section = timeline.loc[(a_index <= timeline.index) & (timeline.index < b_index)]
        ax.plot(none_section.index, none_section.prediction, 'k-', lw=0.15)

        # plot on (insert + None)
        ax.plot(insert.index, insert.prediction, 'k-', lw=0.15)
        ax.plot(insert.index, none_timeline.loc[insert.index].prediction, 'k--', lw=0.15)
        ax.fill_between([insert.index[0], insert.index[-1]],[-0.05,-0.05],[1.05,1.05], color=colour_dict[f'{eruption_num}'], zorder=1, label=f'te={eruption_num}', alpha=0.5)
        try:
            a_index = timeline.loc[insert.index[-1] < timeline.index].index[0]
        except IndexError:
            continue
    # eruptions
    for te in fm.data.tes:
        ax.axvline(te, color='r', linestyle='--', zorder=5, linewidth=0.5)
    ax.axvline(te, color='r', linestyle='--', label='eruption', linewidth=0.5)
    ax.set_xlim(datetimeify('2011-01-01'), datetimeify('2020-01-01'))
    ax.set_ylim(0,1)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
    ax.set_yticks([0,1])
    ax.set_yticklabels([0,1], fontsize=8)
    # ax.legend()
    plt.savefig("fig1.png", format='png', dpi=300)


# Visualisation of unique sigmoid curves + large curve
def fig2():
    ''' Requires construct_hires_timeline() to be run [see calibration.py]

    - uses pp_dir
    - gets the te models using glob
    - for each te model:
        * append the load data frame
        * concatenate the dataframes
        * run _sigmoid_calibration for a,b
    PLOTTING
    - plot each te with above dicts
    - plot overall curve
    - bob's your uncle
    NOTE: this method assumes only one holdout model per fnum
    '''
    eruption_nums = [0,1,2,3,4]
    data_streams = ['rsam', 'mf', 'hf', 'dsar']
    fm = ForecastModel(ti='2011-01-01', tf='2020-01-01', window=2., overlap=0.75,
                       look_forward=2., data_streams=data_streams, root=f'calibration_forecast_model', savefile_type='pkl')
    f_load = f"{fm.rootdir}/calibration/{fm.root}__hires_test__TIMELINE__ncl_500.pkl"
    try:
        timeline = load_dataframe(f_load, index_col='time')
    except FileNotFoundError:
        print("file {f_load} not found, please run construct_hires_timeline() [calibration.py]")

    # Read in timeline a and b from file
    with open(f"{fm.rootdir}/calibration/sigmoid_parameters__hires_test__ncl_500.csv", "r") as f:
        tl_a, tl_b = (float(val) for val in f.readlines()[1].split(','))
    pp_dir = f"{fm.rootdir}/calibration/{fm.root}_hires_predictions__with_insertions"

    fig, ax = plt.subplots(figsize=[8,3])
    fig.tight_layout

    # Plot each singular out of sample curve
    for enum in eruption_nums:
        df_list = list()
        efiles = glob(f'{pp_dir}/*te_{enum}*')
        efiles.sort()
        for fl in efiles:
            df_list.append(load_dataframe(fl))
        enum_timeline = pd.concat(df_list)
        # run sigmoid calibration for enum
        ys = pd.DataFrame(fm._get_label(enum_timeline.index.values),
                        columns=['label'], index=enum_timeline.index)
        a, b = _sigmoid_calibration(enum_timeline.prediction, ys)
        plt.scatter(enum_timeline.prediction, expit(-(a * enum_timeline.prediction + b)),
            s=15, zorder=2, alpha = 0.6, label=f'te={enum}; a={a}, b={b}', color=colour_dict[str(enum)], facecolors='none', linewidth=0.3)

    # Plot the big timeline curve
    plt.scatter(timeline.prediction, expit(-(tl_a * timeline.prediction + tl_b)),
        s=15, zorder=4, alpha = 0.6, label=f'Timeline; a={tl_a}, b={tl_b}', color='black', facecolors='none', linewidth=0.3)

    plt.axvline(0.8, color='pink', linewidth=5, zorder=1, label="Forecast threshold")

    plt.xlabel("Forecast Consensus", fontsize=8)
    plt.ylabel("Calibrated Probability of eruption", fontsize=8)
    plt.title(f"Sigmoids by eruption domain", fontsize=8)
    for t in ax.get_xticklabels() + ax.get_yticklabels():  # increase of x and y tick labels
        t.set_fontsize(6.)

    ax.legend(fontsize=8)
    plt.savefig("fig2.png", format='png', dpi=300)
    plt.close(fig)
def fig2alt():
    ''' Requires construct_hires_timeline() to be run [see calibration.py]

    - uses pp_dir
    - gets the te models using glob
    - for each te model:
        * append the load data frame
        * concatenate the dataframes
        * run _sigmoid_calibration for a,b
    PLOTTING
    - plot each te with above dicts
    - plot overall curve
    - bob's your uncle
    NOTE: this method assumes only one holdout model per fnum
    '''
    eruption_nums = [0,1,2,3,4]
    data_streams = ['rsam', 'mf', 'hf', 'dsar']
    fm = ForecastModel(ti='2011-01-01', tf='2020-01-01', window=2., overlap=0.75,
                       look_forward=2., data_streams=data_streams, root=f'calibration_forecast_model', savefile_type='pkl')
    f_load = f"{fm.rootdir}/calibration/{fm.root}__hires_test__TIMELINE__ncl_500.pkl"
    try:
        timeline = load_dataframe(f_load, index_col='time')
    except FileNotFoundError:
        print("file {f_load} not found, please run construct_hires_timeline() [calibration.py]")

    # Read in timeline a and b from file
    with open(f"{fm.rootdir}/calibration/sigmoid_parameters__hires_test__ncl_500.csv", "r") as f:
        tl_a, tl_b = (float(val) for val in f.readlines()[1].split(','))
    pp_dir = f"{fm.rootdir}/calibration/{fm.root}_hires_predictions__with_insertions"

    fig, ax = plt.subplots(figsize=[8,3])
    fig.tight_layout

    # Plot each singular out of sample curve
    for enum in eruption_nums:
        df_list = list()
        efiles = glob(f'{pp_dir}/*te_{enum}*')
        efiles.sort()
        for fl in efiles:
            df_list.append(load_dataframe(fl))
        enum_timeline = pd.concat(df_list)
        # run sigmoid calibration for enum
        ys = pd.DataFrame(fm._get_label(enum_timeline.index.values),
                        columns=['label'], index=enum_timeline.index)
        a, b = _sigmoid_calibration(enum_timeline.prediction, ys)
        plt.scatter(enum_timeline.prediction, expit(-(a * enum_timeline.prediction + b)),
            s=15, zorder=2, alpha = 0.6, label=f'te={enum}; a={a}, b={b}', color='black', marker=marker_dict[str(enum)], facecolors='none', linewidth=0.3)

    # Plot the big timeline curve
    plt.scatter(timeline.prediction, expit(-(tl_a * timeline.prediction + tl_b)),
        s=15, zorder=4, alpha = 0.6, label=f'Timeline; a={tl_a}, b={tl_b}', color='goldenrod', facecolors='none', linewidth=0.3)

    plt.axvline(0.8, color='pink', linewidth=5, zorder=1, label="Forecast threshold")

    plt.xlabel("Forecast Consensus", fontsize=8)
    plt.ylabel("Calibrated Probability of eruption", fontsize=8)
    plt.title(f"Sigmoids by eruption domain", fontsize=8)
    for t in ax.get_xticklabels() + ax.get_yticklabels():  # increase of x and y tick labels
        t.set_fontsize(6.)

    ax.legend(fontsize=8)
    plt.savefig("fig2alt.png", format='png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    ### --- output vs probability of eruption plots Refer to https://dempseyresearchgroup.slack.com/archives/C01E27W4A5R/p1607477796071700
    # stacked_by_eruption_plot()
    # stacked_by_test_window_plot()
    # stacked_by_all_plot(save=True)

    ### --- timeline plots
    fig1()
    fig2()
    fig2alt()

