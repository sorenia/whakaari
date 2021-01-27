import joblib
import os, sys ### --- added sys
sys.path.insert(0, os.path.abspath('..')) ### --- added line
# os.chdir('..')  # set working directory to root
from glob import glob

import matplotlib.pyplot as plt
from whakaari import TremorData, ForecastModel, save_dataframe, load_dataframe
from datetime import timedelta
import numpy as np
import pandas as pd
pd.plotting.register_matplotlib_converters() # Deal with weird date plotting error
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

''' Import for sigmoid calibration
'''
from sklearn.utils import check_array, indexable, column_or_1d
from scipy.special import expit
from scipy.special import xlogy
from scipy.optimize import fmin_bfgs
from math import log

# new class to mock the forecast_model class
class MockForecastModel(BaseEstimator, ClassifierMixin):

    def __init__(self, model_path: str):
        """
        :param model_path: path to model decision trees.
        """
        self.model_path = model_path
        self.trees = []
        self.feats = []
        self.feats_idx = []

        # classes_ gives the classes in numerical form
        # means that the first column in predict_prob is associated with 0 and second is with 1
        self.classes_ = np.array([0, 1])

        self._read_trees()

    def _read_trees(self):
        # read in the trees and save to self.trees
        tree_files = glob(f"{self.model_path}{os.sep}*.pkl")
        tree_files.sort()
        for tree in tree_files:
            self.trees.append(joblib.load(tree))

        # read in the feature files for each tree
        tree_feat_files = glob(f"{self.model_path}{os.sep}[0-9]*.fts")
        tree_feat_files.sort()
        for feat_file in tree_feat_files:
            f = open(feat_file, 'r')
            feats = list(map(lambda x: x.strip().split(' ', 1)[1], f.readlines()))
            f.close()
            self.feats.append(feats)

    def fit(self, X, y):
        """
        Mock fit function. Does nothing.

        :param X: numpy array of shape [n_samples, n_features]
            Training set.
        :param y:  numpy array of shape [n_samples]
            Target values.
        :return:
        """

        # this class is only for calibration
        # so this is empty
        pass

    def predict(self, X):
        """
        :param X: array-like of shape (n_samples, n_features)
            Test samples.
        :return: y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted labels for X.
        """
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):

        if not self.feats_idx:
            # check to make sure feats_idx array has been initialised
            raise AssertionError("self.feats_idx is empty. Run prepare_for_calibration method.")

        if isinstance(X, pd.DataFrame):
            # convert to 2D numpy array
            X = X.to_numpy()

        # initialise probability matrix
        y_proba = np.zeros((X.shape[0], 2))

        for idx, tree in enumerate(self.trees):
            # use features from feature files when predicting so they match the features used for training
            pred = tree.predict(X[:, self.feats_idx[idx]])

            # turn the predictions into indices 0 and 1 by converting them to integer
            pred = list(map(int, np.round(pred)))
            # add a count to the classification made for each observation
            for i, p in enumerate(pred):
                y_proba[i, p] += 1

        # turning counts into probabilities
        y_proba = y_proba/len(self.trees)

        return y_proba

    def prepare_for_calibration(self, X):
        # save the indices of columns corresponding to the features used in the trees
        self.feats_idx = []
        for features in self.feats:
            self.feats_idx.append([X.columns.get_loc(f) for f in features])

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


def calibration(download_data=False, plots=True, eruption_num=4):
    # constants
    month = timedelta(days=365.25 / 12)

    td = TremorData()
    if download_data:
        for te in td.tes:
            td.update(ti=te-month, tf=te+month)

    # construct model object
    data_streams = ['rsam', 'mf', 'hf', 'dsar']
    fm = ForecastModel(ti='2011-01-01', tf='2020-01-01', window=2., overlap=0.75,
                       look_forward=2., data_streams=data_streams, root=f'calibration_forecast_model', savefile_type='pkl')

    # set modeldir but reuse features in root
    fm.modeldir = f'{fm.modeldir}__te_{eruption_num}'

    # columns to manually drop from feature matrix because they are highly correlated to other
    # linear regressors
    drop_features = ['linear_trend_timewise', 'agg_linear_trend']

    # set the available CPUs higher or lower as appropriate
    n_jobs = 3

    # train the model, excluding 2019 eruption
    # note: building the feature matrix may take several hours, but only has to be done once
    # and will intermittantly save progress in ../features/
    # trained scikit-learn models will be saved to ../models/*root*/
    if eruption_num is not None:
        te = td.tes[eruption_num]
        ti_test = te-month
        tf_test = te+month
        exclude_dates = [[ti_test, tf_test], ]
    else:
        te = None
        ti_test = fm.ti_model
        tf_test = fm.tf_model
        exclude_dates = []
    fm.train(ti='2011-01-01', tf='2020-01-01', drop_features=drop_features, retrain=True,
             exclude_dates=exclude_dates, n_jobs=n_jobs)

    classifier = MockForecastModel(fm.modeldir)

    # get feature and labels for eruption not used in training
    X, y = fm._extract_features(ti=ti_test, tf=tf_test)

    # save the indices of the columns corresponding to features for each tree
    classifier.prepare_for_calibration(X)

    calibrated_classifier = CalibratedClassifierCV(classifier,  method='sigmoid', cv='prefit')
    calibrated_classifier.fit(X, y['label'])

    if plots:
        # ==== plot of calibrated probabilities vs thresholds ====
        f, ax = plt.subplots(1, 1, figsize=(18, 12))

        plt.axvline(0.8, color='pink', linewidth=5, zorder=1)
        plt.scatter(classifier.predict_proba(X)[:, 1], calibrated_classifier.predict_proba(X)[:, 1],
                    c='orange', s=60, zorder=2)

        plt.xlabel("Threshold for predicting eruption", fontsize=25)
        plt.ylabel("Probability of eruption", fontsize=25)
        plt.title(f"Threshold vs probability of eruption {eruption_num}", fontsize=40)
        for t in ax.get_xticklabels() + ax.get_yticklabels():  # increase of x and y tick labels
            t.set_fontsize(20.)

        os.makedirs(fm.plotdir, exist_ok=True)
        plt.savefig(f"{fm.plotdir}/threshold_vs_probability__te_{eruption_num}.png", format='png', dpi=300)
        plt.close()

        # ==== plot of actual target vector, predicted consensus values and calibrated probabilities ====
        forecast = fm.forecast(ti=ti_test, tf=tf_test, recalculate=True)
        f, ax = plt.subplots(3, 1, figsize=(18, 12), sharex="all")

        ax[0].plot(X.index, y['label'], linewidth=3)
        ax[0].axvline(te, color='pink', linewidth=3)
        ax[0].set_ylabel("Actual binary target", fontsize=17)
        ax[1].plot(X.index, forecast['consensus'].loc[X.index], linewidth=3)
        ax[1].axvline(te, color='pink', linewidth=3)
        ax[1].set_ylabel("Prediction Consensus", fontsize=17)
        ax[2].plot(X.index, calibrated_classifier.predict_proba(X)[:, 1], linewidth=3)
        ax[2].axvline(te, color='pink', linewidth=3)
        ax[2].set_ylabel("Calibrated Probability", fontsize=17)
        ax[2].set_xlabel("Time", fontsize=17)
        f.suptitle(f"Actual Target, Prediction Consensus and Calibrated Probabilities, te={eruption_num}", fontsize=32.)
        # increase x and y tick labels
        for t in ax[0].get_yticklabels() + ax[1].get_yticklabels() + ax[2].get_yticklabels() + ax[2].get_xticklabels():
            t.set_fontsize(15.)

        os.makedirs(fm.plotdir, exist_ok=True)
        plt.savefig(f"{fm.plotdir}/actual_predictions_probabilities__te_{eruption_num}.png", format='png', dpi=300)
        plt.close()

    # get full feature and labels for timeline
    X_full, y_full = fm._extract_features(ti=fm.ti_train, tf=fm.tf_train)
    predictions = pd.DataFrame({
        "time": X_full.index,
        "calibrated_prediction": calibrated_classifier.predict_proba(X_full)[:, 1], # Calibrated_prediction is from CCCV
        "prediction": classifier.predict_proba(X_full)[:, 1],
    }).set_index('time')

    # Return for calibrator convenience
    calibrator_dict = {
        'calibrator':   calibrated_classifier,
        'ti_test':      ti_test,
        'tf_test':      tf_test,
        'predictions':  predictions,
    }
    return calibrator_dict


def timeline_calibration():
    '''Script to calibrate over the entire timeline

    1) Looped calls to calibration() to generate cccv for 5 eruptions
    2) Temporarily Store calibrator and test_ti, test_tf
    3) Produce full timeline of predictions [from calibration()]
        a) average where no test data
        b) insert holdout test performance
    4) generate plot?
    '''
     # construct model object FOR ROOTDIR ONLY
    data_streams = ['rsam', 'mf', 'hf', 'dsar']
    fm = ForecastModel(ti='2011-01-01', tf='2020-01-01', window=2., overlap=0.75,
                       look_forward=2., data_streams=data_streams, root=f'calibration_forecast_model', savefile_type='pkl')

    # Generate and store forecast model outputs - Calibrators are there for later comparisons
    calibrators = list()
    for i in range(5):
        calibrator = calibration(eruption_num=i, plots=False)
        f_name = f"{fm.rootdir}/calibration/calibration_forecast_model__te_{i}.pkl"
        save_dataframe(calibrator['predictions'], f_name, index_label = 'time')
        calibrators.append(calibrator)
    calibrator = calibration(eruption_num=None, plots=False)
    f_name = f"{fm.rootdir}/calibration/calibration_forecast_model__te_None.pkl"
    save_dataframe(calibrator['predictions'], f_name, index_label = 'time')
    calibrators.append(calibrator)

    # construct timeline and insert the out of sample predictions
    month = timedelta(days=365.25 / 12)
    f_load = f"{fm.rootdir}/calibration/calibration_forecast_model__te_None.pkl"
    timeline = load_dataframe(f_load, index_col='time')

    for i, te in enumerate(TremorData().tes):
        ti_test = te-month
        tf_test = te+month
        f_load = f"{fm.rootdir}/calibration/calibration_forecast_model__te_{i}.pkl"
        load_df = load_dataframe(f_load, index_col='time')
        out_of_sample = load_df.loc[(load_df.index >= ti_test) & (load_df.index < tf_test)]

        # Update seems simple enough https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.update.html
        timeline.update(out_of_sample)

    # run sigmoid calibration
    ys = pd.DataFrame(fm._get_label(timeline.index.values), columns=['label'], index=timeline.index)
    a,b = _sigmoid_calibration(timeline.prediction, ys)
    with open(f"{fm.rootdir}/calibration/sigmoid_parameters__full_calibrated.csv", "w") as f:
        f.write(f"a,b\n{a},{b}")

    # and apply function
    timeline['full_calibrated'] = timeline['prediction'].apply(get_calibrated, a=a, b=b)
    timeline['ys'] = ys

    f_save = f"{fm.rootdir}/calibration/{fm.root}__TIMELINE.pkl"
    save_dataframe(timeline, f_save, index_label='time')

    # ==== plot of calibrated probabilities vs thresholds ====
    f, ax = plt.subplots(1, 1, figsize=(18, 12))

    plt.axvline(0.8, color='pink', linewidth=5, zorder=1)
    plt.scatter(timeline.prediction, timeline.full_calibrated, c='orange', s=50, zorder=2)
    plt.xlabel("Timeline probability output", fontsize=25)
    plt.ylabel("Calibrated Sigmoid Output", fontsize=25)
    plt.title(f"Threshold vs probability full timeline", fontsize=28)
    for t in ax.get_xticklabels() + ax.get_yticklabels():  # increase of x and y tick labels
        t.set_fontsize(20.)

    os.makedirs(fm.plotdir, exist_ok=True)
    plt.savefig(f"{fm.plotdir}/threshold_vs_probability__timeline.png", format='png', dpi=300)
    plt.close()


def _sigmoid_calibration(df, y, sample_weight=None):
    """Probability Calibration with sigmoid method (Platt 2000)
    See sklearn.calibration module L392-452
    https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/calibration.py#L392
    Parameters
    ----------
    df : ndarray, shape (n_samples,)
        The decision function or predict proba for the samples.
    y : ndarray, shape (n_samples,)
        The targets.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If None, then samples are equally weighted.
    Returns
    -------
    a : float
        The slope.
    b : float
        The intercept.
    References
    ----------
    Platt, "Probabilistic Outputs for Support Vector Machines"
    """
    df = column_or_1d(df)
    y = column_or_1d(y)

    F = df  # F follows Platt's notations

    # Bayesian priors (see Platt end of section 2.2)
    prior0 = float(np.sum(y <= 0))
    prior1 = y.shape[0] - prior0
    T = np.zeros(y.shape)
    T[y > 0] = (prior1 + 1.) / (prior1 + 2.)
    T[y <= 0] = 1. / (prior0 + 2.)
    T1 = 1. - T

    def objective(AB):
        # From Platt (beginning of Section 2.2)
        P = expit(-(AB[0] * F + AB[1]))
        loss = -(xlogy(T, P) + xlogy(T1, 1. - P))
        if sample_weight is not None:
            return (sample_weight * loss).sum()
        else:
            return loss.sum()

    def grad(AB):
        # gradient of the objective function
        P = expit(-(AB[0] * F + AB[1]))
        TEP_minus_T1P = T - P
        if sample_weight is not None:
            TEP_minus_T1P *= sample_weight
        dA = np.dot(TEP_minus_T1P, F)
        dB = np.sum(TEP_minus_T1P)
        return np.array([dA, dB])

    AB0 = np.array([0., log((prior0 + 1.) / (prior1 + 1.))])
    AB_ = fmin_bfgs(objective, AB0, fprime=grad, disp=False)
    return AB_[0], AB_[1]

def get_calibrated(prediction, a, b):
    ''' Helper function to apply a,b to set of predictions (see pd.DataFrame.apply())
    '''
    return expit(-(a * prediction + b))


def get_alertdays(alerts, lf):
    ''' Helper function to convert model alerts into alert days

        Parameters:
        -----------
        alerts : pd.Series / column of pd.DataFrame
            The periods when the model gives an alerts. NOTE assumes index is of Datetime
        lf : float
            Days for lookforward alert (Default 2.) Used to construct ys

        Returns:
        --------
        alert_days: pd.Series
            thresholds used for calculating
    '''
    alert_period = timedelta(days=lf)
    alert_days = alerts.copy()
    als = alert_days.loc[alert_days == 1]
    for al in als.index:
        start = al
        alert_days.loc[start:start+alert_period] = 1
    return alert_days


def get_falsealerts(alerts, lf, tes):
    ''' Helper function to convert model alerts into false alert ratio

        This function assumes that alerts start on the time issued and end prior to the look forward

        Parameters:
        -----------
        alerts : pd.Series / column of pd.DataFrame
            The periods when the model gives an alerts. NOTE assumes index is of Datetime
        lf : float
            Days for lookforward alert (Default 2.) Used to construct ys
        tes : list of Datetimes
            Start of eruptions

        Returns:
        --------
        falsealert_ratio: float
            ratio of false alerts : true alerts a.k.a. false alerts/all alerts
    '''
    alert_period = timedelta(days=lf)
    modelalerts = alerts.loc[alerts == 1]
    if len(modelalerts) == 0:
        return 0
    falsealerts = 0
    for al in modelalerts.index:
        eruption=False
        for te in tes:
            if al <= te and te < al+alert_period:
                eruption=True
                break
        if not eruption: falsealerts=falsealerts+1

    falsealert_ratio = falsealerts/len(modelalerts)
    return falsealert_ratio


def get_truealerts(alerts, lf, tes):
    ''' Handle function that returns true alarms a.k.a. 1-falsealarm_ratio
    '''
    return 1-get_falsealerts(alerts, lf, tes)


def construct_timeline():
    data_streams = ['rsam', 'mf', 'hf', 'dsar']
    fm = ForecastModel(ti='2011-01-01', tf='2020-01-01', window=2., overlap=0.75,
                       look_forward=2., data_streams=data_streams, root=f'calibration_forecast_model', savefile_type='pkl')

    try:
        f_load = f"{fm.rootdir}/calibration/{fm.root}__TIMELINE.pkl"
        timeline = load_dataframe(f_load, index_col='time')
        return timeline
    except FileNotFoundError:
        print(f"file {f_load} not found... constructing timeline")

    # construct timeline and insert the out of sample predictions
    month = timedelta(days=365.25 / 12)
    f_load = f"{fm.rootdir}/calibration/{fm.root}__te_None.pkl"
    timeline = load_dataframe(f_load, index_col='time')

    for i, te in enumerate(TremorData().tes):
        ti_test = te-month
        tf_test = te+month
        f_load = f"{fm.rootdir}/calibration/{fm.root}__te_{i}.pkl"
        load_df = load_dataframe(f_load, index_col='time')
        out_of_sample = load_df.loc[(
            load_df.index >= ti_test) & (load_df.index < tf_test)]

        # Update seems simple enough https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.update.html
        timeline.update(out_of_sample)

    # run sigmoid calibration
    ys = pd.DataFrame(fm._get_label(timeline.index.values),
                      columns=['label'], index=timeline.index)
    a, b = _sigmoid_calibration(timeline.prediction, ys)
    with open(f"{fm.rootdir}/calibration/sigmoid_parameters__full_calibrated.csv", "w") as f:
        f.write(f"a,b\n{a},{b}")

    timeline['full_calibrated'] = timeline['prediction'].apply(
        get_calibrated, a=a, b=b)
    timeline['ys'] = ys

    f_save = f"{fm.rootdir}/calibration/{fm.root}__TIMELINE.pkl"
    save_dataframe(timeline, f_save, index_label='time')
    return timeline


def single_sweep(pp, ys, tes, lf=2., thresholds=[0.005, 0.05], inplace=False):
    ''' This function takes a timeline of predict_proba(), ys and probability thresholds

        Parameters:
        -----------
        pp : pandas.DataFrame
            Dataframe of windowed predict_proba()
        ys : ndarray, shape (n_samples,)
            The targets.
        tes : list of Datetime
            Times when eruptions occur
        lf : float
            Days for lookforward alert (Default 2.) Used to construct ys
        thresholds : list
            List of floats to conduct thresholds (Default range 0.5%-5%)
        inplace: bool
            If the function should make a copy of dataframe

        Returns:
        --------
        thresholds: list
            thresholds used for calculating
        alertday_ratios: list
            alert ratio corresponding to time model in alert
        falsealert_ratios: list
            falsealert ratio corresponding to each alert
        pp : pandas.DataFrame
            Dataframe of windowed data, with 'id' column denoting individual windows.
    '''
    if not inplace: pp = pp.copy() # Copy to make sure the user does not want the dataframe to change

    # Generate a, b from sigmoid calibration
    a, b = _sigmoid_calibration(pp.prediction, ys)

    cal_string = f'calibrated__lf_{lf}'
    # and apply sigmoid function
    pp[cal_string] = pp['prediction'].apply(get_calibrated, a=a, b=b)

    # if 2 numbers in list, assume start and end point
    if len(thresholds) == 2:
        thresholds = np.round(np.linspace(
            thresholds[0], thresholds[-1], num=10, endpoint=True), 4)

    alertday_ratios = list()
    accuracies = list()
    falsealert_ratios = list()
    len_tes = len(tes)
    # for each threshold test the calibrated and generate alerts
    for th in thresholds:
        # alert mask
        al_string = f'alerts__lf_{lf}__th_{th}'
        alerts = pp[cal_string] >= th
        pp[al_string] = alerts.astype(int)

        # alert day mask
        ald_string = f'alert_days__lf_{lf}__th_{th}'
        pp[ald_string] = get_alertdays(pp[al_string], lf)

        # Calculation of ratios and accuracies
        alertday_ratios.append(pp[ald_string].sum() / pp[ald_string].count())

        # calculate accuracy here by looping through tes and incrementing count
        correct = 0
        for te in tes:
            last_alert = pp.loc[pp.index<=te][ald_string].iloc[-1]
            if last_alert == 1: correct = correct+1
        accuracies.append(correct/len_tes)

        # Calculate false alarm rate
        falsealert_ratios.append(get_falsealerts(pp[al_string], lf, tes))

    # NOTE: considering switching the return statement to a single dict + pp (if required)
    if inplace:
        return alertday_ratios, accuracies, falsealert_ratios, thresholds
    else:
        return alertday_ratios, accuracies, falsealert_ratios, thresholds, pp


def full_sweep(load_adr=None, load_acc=None, load_far=None):
    ''' This function does every sweep of lookforwards and probability thresholds

    Generates heatmap of lookforwards and probability thresholds
    Does multiple calls to single_sweep() for each lookforward
    Saves the outputs from each sweep into csv file

    This function is also where you sat
    '''
    if load_adr is not None and load_acc is not None and load_far is not None:
        # load files from dir
        adr_df = load_dataframe(load_adr, index_col="thresholds")
        acc_df = load_dataframe(load_acc, index_col="thresholds")
        far_df = load_dataframe(load_far, index_col="thresholds")

    else:
        tes_pop = TremorData().tes
        tes_pop.pop(3) # remove hard earthquake
        timeline = construct_timeline()
        pp = timeline.drop(['ys', 'full_calibrated', 'calibrated_prediction'], axis='columns')

        thresholds = np.round(np.linspace(
            0.005, 0.05, num=10, endpoint=True), 4)
        look_forwards = np.arange(1,7.5, step=0.5)
        alertday_ratios = dict()
        accuracies = dict()
        falsealert_ratios = dict()
        for lf in look_forwards:
            print(f"creating forecast model with lf = {lf}")
            fm = ForecastModel(ti='2011-01-01', tf='2020-01-01', window=2., overlap=0.75,
                            look_forward=lf, root=f'calibration_forecast_model', savefile_type='pkl')
            ys = pd.DataFrame(fm._get_label(pp.index.values), columns=['label'], index=pp.index)
            lf_alertday_ratios, lf_accuracies, lf_falsealert_ratios, _ = single_sweep(pp, ys, tes=tes_pop, lf=lf, thresholds=thresholds, inplace=True)
            alertday_ratios[lf] = lf_alertday_ratios
            accuracies[lf] = lf_accuracies
            falsealert_ratios[lf] = lf_falsealert_ratios
            print(f"done")
        adr_df = pd.DataFrame(alertday_ratios,
                            index=[f'threshold_{th}'for th in thresholds]).add_prefix('lookforward_')
        adr_df.index.name = "thresholds"
        acc_df = pd.DataFrame(accuracies,
                            index=[f'threshold_{th}'for th in thresholds]).add_prefix('lookforward_')
        acc_df.index.name = "thresholds"
        far_df = pd.DataFrame(falsealert_ratios,
                            index=[f'threshold_{th}'for th in thresholds]).add_prefix('lookforward_')
        far_df.index.name = "thresholds"
        save_adr = f"{fm.rootdir}/calibration/heatmap/alertdayratios_df.csv"
        save_acc = f"{fm.rootdir}/calibration/heatmap/accuracies_df.csv"
        save_far = f"{fm.rootdir}/calibration/heatmap/falsealertratios_df.csv"
        save_dataframe(adr_df, save_adr)
        save_dataframe(acc_df, save_acc)
        save_dataframe(far_df, save_far)
    return adr_df, acc_df, far_df

def plot_heatmap():
    ''' This function calls full_sweep() with saved dataframes then creates heatmap
    '''
    fm = ForecastModel(ti='2011-01-01', tf='2020-01-01', window=2., overlap=0.75,
                       look_forward=2., root=f'calibration_forecast_model', savefile_type='pkl')
    tes_pop = TremorData().tes
    tes_pop.pop(3) # remove hard earthquake
    load_adr = f"{fm.rootdir}/calibration/heatmap/heatmap_df.csv"
    load_acc = f"{fm.rootdir}/calibration/heatmap/accuracies_df.csv"
    adr_df, acc_df = full_sweep(load_adr, load_acc)
    acc_df = acc_df*len(tes_pop)
    acc_df = acc_df.astype(int)
    # colours here
    cmap_dict = {
        "1": "Reds",
        "2": "Oranges",
        "3": "Blues",
        "4": "Greens",
    }

    fig, ax = plt.subplots()

    # extent sets the axis ticks x_left,x_right,y_bottom,y_top
    ax.imshow(acc_df.where(acc_df == 0), cmap="Greys", interpolation=None,
              vmin=0, vmax=1, extent=[0, 14, 0.05, 0], aspect='auto')
    for acc in cmap_dict.keys():
        adr_map = 1-adr_df.where(acc_df==int(acc))
        ax.imshow(adr_map, cmap=cmap_dict[acc],
                  interpolation=None, vmin=0, vmax=1, extent=[0.5, 13.5, 0.05, 0.005],aspect='auto')
    plt.ylabel('probability thresholds for alert')
    plt.xlabel('look_forward (days)')
    plt.title("Heatmap showing alert day ratio given probability and lookforward")
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.show()


def plot_contours():
    ''' This function calls full_sweep() with saved dataframes then creates contour plot
    '''
    fm = ForecastModel(ti='2011-01-01', tf='2020-01-01', window=2., overlap=0.75,
                       look_forward=2., root=f'calibration_forecast_model', savefile_type='pkl')
    tes_pop = TremorData().tes
    tes_pop.pop(3) # remove hard earthquake
    load_adr = f"{fm.rootdir}/calibration/heatmap/heatmap_df.csv"
    load_acc = f"{fm.rootdir}/calibration/heatmap/accuracies_df.csv"
    load_far = f"{fm.rootdir}/calibration/heatmap/falsealertratios_df.csv"
    # adr_df, acc_df, far_df = full_sweep(
    #     load_adr=load_adr, load_acc=load_acc, load_far=load_far)
    adr_df, acc_df, far_df = full_sweep()
    acc_df = acc_df*len(tes_pop)
    acc_df = acc_df.astype(int)
    # colours here
    cmap_dict = {
        "0": "tab:gray",
        "1": "tab:red",
        "2": "tab:orange",
        "3": "tab:blue",
        "4": "tab:green",
    }

    clist= [v for k,v in cmap_dict.items()]

    fig, axs = plt.subplots(1,2,figsize=(10.5,18.5/2),sharey=True)
    col_names = adr_df.columns.values
    col_names = [x.split('_')[-1] for x in col_names]
    row_names = adr_df.index.values
    row_names = [y.split('_')[-1] for y in row_names]
    z = acc_df.values
    for ax in axs:
        ct = ax.contourf(col_names, row_names, z, colors=clist, levels=[i-.5 for i in range(6)], alpha=0.8)
        ax.tick_params(labelsize=8)
        ax.tick_params(axis='x', labelrotation=0.25)
        ax.set_xlabel('Lookforwards', fontsize=12)

    axs[0].set_title('Alert Day Ratio', fontsize=24)
    axs[0].set_ylabel('Thresholds', fontsize=12)
    axs[1].set_title('False Alert Ratio', fontsize=24)
    fig.colorbar(ct)
    # fig.set_size_inches(18.5, 10.5)
    plt.show()


if __name__ == '__main__':
    # os.chdir('..')  # set working directory to root
    # calibration()
    # timeline_calibration()
    # full_sweep()
    # plot_heatmap()
    plot_contours()
