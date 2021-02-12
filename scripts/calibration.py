import joblib
import os, sys ### --- added sys
sys.path.insert(0, os.path.abspath('..')) ### --- added line
# os.chdir('..')  # set working directory to root
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from whakaari import TremorData, ForecastModel, save_dataframe, load_dataframe, makedir, datetimeify
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

# For progress bar
from tqdm import tqdm

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


def calibration(download_data=False, plots=True, eruption_num=4, ncl=100):
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
    fm.modeldir = f'{fm.modeldir}__te_{eruption_num}__ncl_{ncl}'

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
             exclude_dates=exclude_dates, n_jobs=n_jobs, Ncl=ncl)

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
        "calibrated_prediction": calibrated_classifier.predict_proba(X_full)[:, 1], # Calibrated_prediction is from CCCV NOTE: unused so could skip CCCV()
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


def timeline_calibration(ncl=100):
    '''Script to calibrate over the entire timeline

    1) Looped calls to calibration() to generate cccv for 5 eruptions
    2) Temporarily Store calibrator and test_ti, test_tf
    3) Produce full timeline of predictions [from calibration()]
        a) te=None where no test data
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
        calibrator = calibration(eruption_num=i, plots=False, ncl=ncl)
        f_name = f"{fm.rootdir}/calibration/calibration_forecast_model__te_{i}__ncl_{ncl}.pkl"
        save_dataframe(calibrator['predictions'], f_name, index_label = 'time')
        calibrators.append(calibrator)
    calibrator = calibration(eruption_num=None, plots=False, ncl=ncl)
    f_name = f"{fm.rootdir}/calibration/calibration_forecast_model__te_None__ncl_{ncl}.pkl"
    save_dataframe(calibrator['predictions'], f_name, index_label = 'time')
    calibrators.append(calibrator)

    # construct timeline and insert the out of sample predictions
    month = timedelta(days=365.25 / 12)
    f_load = f"{fm.rootdir}/calibration/calibration_forecast_model__te_None__ncl_{ncl}.pkl"
    timeline = load_dataframe(f_load, index_col='time')

    for i, te in enumerate(TremorData().tes):
        ti_test = te-month
        tf_test = te+month
        f_load = f"{fm.rootdir}/calibration/calibration_forecast_model__te_{i}__ncl_{ncl}.pkl"
        load_df = load_dataframe(f_load, index_col='time')
        out_of_sample = load_df.loc[(load_df.index >= ti_test) & (load_df.index < tf_test)]

        # Update seems simple enough https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.update.html
        timeline.update(out_of_sample)
    # Run sigmoid calibration with a set lookforward of 5 for a and b
    fm_5 = ForecastModel(ti='2011-01-01', tf='2020-01-01', window=2., overlap=0.75,
                       look_forward=5., data_streams=data_streams, root=f'calibration_forecast_model', savefile_type='pkl')
    # run sigmoid calibration
    ys = pd.DataFrame(fm_5._get_label(timeline.index.values), columns=['label'], index=timeline.index)
    a,b = _sigmoid_calibration(timeline.prediction, ys)
    with open(f"{fm.rootdir}/calibration/sigmoid_parameters__full_calibrated.csv", "w") as f:
        f.write(f"a,b\n{a},{b}")

    # and apply function
    timeline['full_calibrated'] = timeline['prediction'].apply(get_calibrated, a=a, b=b)
    timeline['ys'] = ys

    f_save = f"{fm.rootdir}/calibration/{fm.root}__TIMELINE__ncl_{ncl}.pkl"
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
    plt.savefig(f"{fm.plotdir}/threshold_vs_probability__timeline__ncl_{ncl}.png", format='png', dpi=300)
    plt.close()
    return timeline


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


def get_alertdays(alerts, lf, alertwindow=True):
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
    modelalerts_time = alerts.loc[alerts == 1].index
    alert_days = alerts.copy()

    if len(modelalerts_time) == 0:
        return 0

    if alertwindow:
        # david version of model alerts
        # Create non-overlapping alert windows as list of (ti_j, tf_j+alert) for j in non overlapping
        aw_ap = np.array([alert_period],dtype='timedelta64')[0] # Alert Window Alert Period
        non_op_inds = np.where(np.diff(modelalerts_time)>aw_ap)[0]
        alert_windows = list(zip(
            [modelalerts_time[0],] +
            [modelalerts_time[i+1] for i in non_op_inds],
            [modelalerts_time[j] + aw_ap for j in non_op_inds] +
            [modelalerts_time[-1] + aw_ap]
        ))

        # Construct alerts np.array
        ad_inds = [alert_days.loc[aw[0]:aw[1]].index for aw in alert_windows]
        alert_days.loc[np.concatenate(ad_inds)] = 1
    else:
        for al in modelalerts_time:
            start = al
            alert_days.loc[start:start+alert_period] = 1
    return alert_days


def get_falsealerts(alerts, lf, tes, alertwindow=True):
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
    modelalerts_ind = alerts.loc[alerts == 1].index
    if len(modelalerts_ind) == 0:
        return 0

    falsealerts = 0
    if alertwindow:
        # david version of model alerts
        # Create non-overlapping alert windows as list of (ti_j, tf_j+alert) for j in non overlapping
        aw_ap = np.array([alert_period],dtype='timedelta64')[0] # Alert Window Alert Period
        non_op_inds = np.where(np.diff(modelalerts_ind)>aw_ap)[0]
        alert_windows = list(zip(
            [modelalerts_ind[0],] +
            [modelalerts_ind[i+1] for i in non_op_inds],
            [modelalerts_ind[j] + aw_ap for j in non_op_inds] +
            [modelalerts_ind[-1] + aw_ap]
        ))

        # Check for eruption detected
        for aw in alert_windows:
            eruption = False
            for te in tes:
                if aw[0] <= te and te < aw[1]:
                    eruption = True
                    break
            if not eruption: falsealerts += 1
        falsealert_ratio = falsealerts/len(alert_windows)
    else:
        # Stephen Version of model alerts
        for al in modelalerts_ind:
            eruption=False
            for te in tes:
                if al <= te and te < al+alert_period:
                    eruption=True
                    break
            if not eruption: falsealerts+=1

        falsealert_ratio = falsealerts/len(modelalerts_ind)
    return falsealert_ratio


def get_truealerts(alerts, lf, tes):
    ''' Handle function that returns true alarms a.k.a. 1-falsealarm_ratio
    '''
    return 1-get_falsealerts(alerts, lf, tes)


def compute_model_alerts(alerts, lf, tes):
    alert_period = timedelta(days=lf)
    modelalerts_time = alerts.loc[alerts == 1].index
    if len(modelalerts_time) == 0:
        model_alerts = {
            'false_alert' : 0,
            'missed' : len(tes),
            'true_alert' : 0,
            'true_negative' : int(1e8),
            'dur' : 0,
            'mcc' : 0,
        }
        return model_alerts

    falsealerts = 0

    # Create non-overlapping alert windows as list of (t0_j, t1_j+alert) for j in non overlapping
    aw_ap = np.array([alert_period],dtype='timedelta64')[0] # Alert Window Alert Period
    non_op_inds = np.where(np.diff(modelalerts_time)>aw_ap)[0]
    alert_windows = list(zip(
        [modelalerts_time[0],] +
        [modelalerts_time[i+1] for i in non_op_inds],
        [modelalerts_time[j] + aw_ap for j in non_op_inds] +
        [modelalerts_time[-1] + aw_ap]
    ))

    alert_window_lengths = [np.diff(aw) for aw in alert_windows]
    pop_tes = tes.copy()
    true_alert = 0
    false_alert = 0
    inalert = 0.
    missed = 0
    total_time = (alerts.index[-1] - alerts.index[0]).total_seconds()

    # dti = timedelta(days=(1-self.overlap)*self.window) # _model_alerts uses dti to distinguish hires/lores

    # Check for eruption detected
    for t0, t1 in alert_windows:

        inalert += ((t1-t0)).total_seconds() # fm._model_alerts() uses indices, here uses datetime directly
        # no eruptions left to classify, only misclassifications now
        if len(pop_tes) == 0:
            false_alert += 1
            continue

        # eruption has been missed
        while pop_tes[0] < t0:
            pop_tes.pop(0)
            missed += 1
            if len(pop_tes) == 0:
                break
        if len(pop_tes) == 0: # Continue loop after popping the final eruption in list
            continue

        # alert does NOT contain any eruption, move to next alert window
        if not (t0 < pop_tes[0]  and pop_tes[0] <= t1):
            false_alert += 1
            continue

        # alert window contains eruption(s), check for more eruptions, else move to next alert window
        while pop_tes[0] > t0 and pop_tes[0] <= t1:
            pop_tes.pop(0)
            true_alert += 1
            if len(pop_tes) == 0:
                break
    # any remaining eruptions after alert windows must have been missed
    missed += len(tes)

    model_alerts = {
        'false_alert' : false_alert,
        'missed' : missed,
        'true_alert' : true_alert,
        # 'true_negative' : int((len(alerts)-np.sum(alert_window_lengths))/np.mean(alert_window_lengths))-missed,
        'dur' : inalert/total_time,
        # 'mcc' : mcc,
    }
    return model_alerts


def construct_timeline(ncl=100):
    data_streams = ['rsam', 'mf', 'hf', 'dsar']
    fm = ForecastModel(ti='2011-01-01', tf='2020-01-01', window=2., overlap=0.75,
                       look_forward=2., data_streams=data_streams, root=f'calibration_forecast_model', savefile_type='pkl')

    try:
        f_load = f"{fm.rootdir}/calibration/{fm.root}__TIMELINE__ncl_{ncl}.pkl"
        timeline = load_dataframe(f_load, index_col='time')
        return timeline
    except FileNotFoundError:
        print(f"file {f_load} not found... constructing timeline")

    # construct TIMELINE and insert the out of sample predictions NOTE: Assumes timeline_calibration() has been run
    try:
        f_load = f"{fm.rootdir}/calibration/{fm.root}__te_None__ncl_{ncl}.pkl"
        timeline = load_dataframe(f_load, index_col='time')
    except FileNotFoundError:
        print(f"file {f_load} not found... constructing forecast models from timeline_calibration()")
        timeline = timeline_calibration(ncl=ncl)

    month = timedelta(days=365.25 / 12)
    for i, te in enumerate(TremorData().tes):
        ti_test = te-month
        tf_test = te+month
        f_load = f"{fm.rootdir}/calibration/{fm.root}__te_{i}__ncl_{ncl}.pkl"
        load_df = load_dataframe(f_load, index_col='time')
        out_of_sample = load_df.loc[(
            load_df.index >= ti_test) & (load_df.index < tf_test)]

        # Update seems simple enough https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.update.html
        timeline.update(out_of_sample)

    # run sigmoid calibration
    ys = pd.DataFrame(fm._get_label(timeline.index.values),
                      columns=['label'], index=timeline.index)
    a, b = _sigmoid_calibration(timeline.prediction, ys)
    with open(f"{fm.rootdir}/calibration/sigmoid_parameters__full_calibrated__ncl_{ncl}.csv", "w") as f:
        f.write(f"a,b\n{a},{b}")

    timeline['full_calibrated'] = timeline['prediction'].apply(
        get_calibrated, a=a, b=b)
    timeline['ys'] = ys

    f_save = f"{fm.rootdir}/calibration/{fm.root}__TIMELINE__ncl_{ncl}.pkl"
    save_dataframe(timeline, f_save, index_label='time')
    return timeline


def construct_test_dates():
    ''' Easily create test dates for each side of the eruption

    returns list of dicts
    '''
    # Initialise list of te and Tremor dates
    tes = TremorData().tes
    month = timedelta(days=365.25 / 12)
    exclude_dates = list()
    for te in tes:
        exclude_dates.append({'ti':te-month, 'tf':te+month})
    return exclude_dates


def which_eruption(start, end, test_times=construct_test_dates()):
        '''takes start/end as datetime and list of test_times constructed using construct_test_dates()

        returns eruption value in eruption_nums
        '''
        # Biased towards earlier eruptions
        for i, t_time in enumerate(test_times):
            # Check if start time falls within any of the testing period
            if t_time['ti'] <= start < t_time['tf']:
                return i
            # Check if end time falls within any of the testing period
            if t_time['ti'] <= end < t_time['tf']:
                return i
        return None


def construct_hires_timeline(ncl=100, n_jobs=3):
    '''
    Pseudocode:
    1. Train ALL models with 500 classifiers on the lores data + Create MockForecastModels (for mockfm.predict_proba())
    2. for week in weeks: generate hires forecast for week using model based on first time on index named hires_forecast_weekX
    3. Concatenate all weekly forecasts
    4. Calibrate the forecasts <= NOT NEEDED see single_sweep()
    '''

    month = timedelta(days=365.25 / 12)
    data_streams = ['rsam', 'mf', 'hf', 'dsar']
    fm = ForecastModel(ti='2011-01-01', tf='2020-01-01', window=2., overlap=0.75,
                       look_forward=2., data_streams=data_streams, root=f'calibration_forecast_model', savefile_type='pkl')
    f_load = f"{fm.rootdir}/calibration/{fm.root}__hires_test__TIMELINE__ncl_{ncl}.pkl"
    if os.path.isfile(f_load):
        return load_dataframe(f_load, index_col='time')
    # columns to manually drop from feature matrix because they are highly correlated to other
    # linear regressors
    drop_features = ['linear_trend_timewise', 'agg_linear_trend']

    # Train ALL models
    td = TremorData()
    eruption_nums = [None, 0, 1, 2, 3, 4]
    for enum in eruption_nums:
        # Setting exclude dates
        if enum is None: # Rearranged for readability
            te = None # Initialised in calibration() for plotting
            ti_test = fm.ti_model
            tf_test = fm.tf_model
            exclude_dates = [] # No dates to exclude
        else:
            te = td.tes[enum] # Initialised in calibration() for plotting
            ti_test = te-month
            tf_test = te+month
            exclude_dates = [[ti_test, tf_test], ]
        # initialise _fm and set model dir
        _fm = ForecastModel(ti='2011-01-01', tf='2020-01-01', window=2., overlap=0.75,
                       look_forward=2., data_streams=data_streams, root=f'calibration_forecast_model', savefile_type='pkl')
        _fm.modeldir = f'{fm.modeldir}__te_{enum}__ncl_{ncl}' # Based off above fm
        # Train fm on lores data no exclude dates
        _fm.train(ti='2011-01-01', tf='2020-01-01', drop_features=drop_features, retrain=False,
                exclude_dates=exclude_dates, n_jobs=n_jobs, Ncl=ncl)

    # Initialise the hires feature paths
    feature_paths = glob(f'{fm.featdir}/{fm.root}_hires__fnum_*_features.pkl')
    feature_paths.sort()

    # Initialise ALL MockForecastModel() for the convenient .predict_proba() // Consider combining with above loop
    classifiers = dict()
    for enum in eruption_nums:
        classifiers[enum] = MockForecastModel(f'{fm.modeldir}__te_{enum}__ncl_{ncl}')

    pp_dir = f"{fm.rootdir}/calibration/{fm.root}_hires_predictions__with_insertions"
    makedir(pp_dir)
    df_list = list()
    test_range = construct_test_dates()
    for feat_file in feature_paths:
        # get hires features and labels for eruption not used in training
        X = load_dataframe(feat_file)
        fnum = int(feat_file.split("fnum_")[-1].split('_features.pkl')[0])

        # Check here for which eruption model to use using t0 and t-1 of X index
        model = which_eruption(X.index[0], X.index[-1], test_range)

        # save the indices of the columns corresponding to features for each tree
        classifiers[model].prepare_for_calibration(X)

        # Calculate raw predict_proba and take the 1 column
        pp = classifiers[model].predict_proba(X)[:,1]

        # make a dataframe (could be slow and have speed improvements later)
        df = pd.DataFrame({'prediction': pp,}, index=X.index)
        f_save = f"{pp_dir}/fnum_{fnum:03}__MODEL_te_{model}.pkl"
        save_dataframe(df, f_save)
        df_list.append(df)

    timeline = pd.concat(df_list)
    f_save = f"{fm.rootdir}/calibration/{fm.root}__hires_test__TIMELINE__ncl_{ncl}.pkl"
    save_dataframe(timeline, f_save, index_label='time')

    # run base sigmoid calibration
    ys = pd.DataFrame(fm._get_label(timeline.index.values),
                      columns=['label'], index=timeline.index)
    a, b = _sigmoid_calibration(timeline.prediction, ys)
    with open(f"{fm.rootdir}/calibration/sigmoid_parameters__hires_test__ncl_{ncl}.csv", "w") as f:
        f.write(f"a,b\n{a},{b}")
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
    # and apply sigmoid function // Could potentially be slowing down performance if i append to dataframe
    # pp[cal_string] = pp['prediction'].apply(get_calibrated, a=a, b=b)
    cal_pp = pp['prediction'].apply(get_calibrated, a=a, b=b)

    # if 2 numbers in list, assume start and end point
    if len(thresholds) == 2:
        thresholds = np.round(np.linspace(
            thresholds[0], thresholds[-1], num=10, endpoint=True), 4)

    alertday_ratios = np.zeros(len(thresholds))
    accuracies = np.zeros(len(thresholds))
    falsealert_ratios = np.zeros(len(thresholds))
    len_tes = len(tes)
    # for each threshold test the calibrated and generate alerts
    for i,th in enumerate(thresholds):
        # alert mask
        al_string = f'alerts__lf_{lf}__th_{th}'
        # alerts = pp[cal_string] >= th
        # pp[al_string] = alerts.astype(int) # // Could potentially be slowing down performance if i append to dataframe
        alerts = cal_pp >= th
        alerts = alerts.astype(int) # // Could potentially be slowing down performance if i append to dataframe

        # # alert day mask
        # ald_string = f'alert_days__lf_{lf}__th_{th}'
        # pp[ald_string] = get_alertdays(pp[al_string], lf)

        # # Calculation of ratios and accuracies
        # alertday_ratios.append(pp[ald_string].sum() / pp[ald_string].count())

        # # calculate accuracy here by looping through tes and incrementing count
        # correct = 0
        # for te in tes:
        #     last_alert = pp.loc[pp.index<=te][ald_string].iloc[-1]
        #     if last_alert == 1: correct = correct+1
        # accuracies.append(correct/len_tes)

        # # Calculate false alarm rate
        # falsealert_ratios.append(get_falsealerts(pp[al_string], lf, tes))

        # MODEL ALERTS
        # ma = compute_model_alerts(pp[al_string], lf, tes)
        ma = compute_model_alerts(alerts, lf, tes)
        alertday_ratios[i] = ma['dur']
        accuracies[i] = ma['true_alert']/len(tes)
        try:
            falsealert_ratios[i] = ma['false_alert'] / (ma['false_alert'] + ma['true_alert'])
        except ZeroDivisionError: # No alerts made give zero division error -> Used falsealert_ratio = 1 for smoother plotting
            falsealert_ratios[i]=1
            continue


    # NOTE: considering switching the return statement to a single dict + pp (if required)
    if inplace:
        return alertday_ratios, accuracies, falsealert_ratios, thresholds
    else:
        return alertday_ratios, accuracies, falsealert_ratios, thresholds, pp


def full_sweep(loaddir=None, ncl=100, savedir=None, hires=False):
    ''' This function does every sweep of lookforwards and probability thresholds

    Generates heatmap of lookforwards and probability thresholds
    Does multiple calls to single_sweep() for each lookforward
    Saves the outputs from each sweep into csv file

    This function is also where you sat
    '''
    if loaddir is not None:
        try: # load files from dir
            load_adr = f"{loaddir}/alertdayratios_df__ncl_{ncl}.csv"
            load_acc = f"{loaddir}/accuracies_df__ncl_{ncl}.csv"
            load_far = f"{loaddir}/falsealertratios_windows_df__ncl_{ncl}.csv"
            adr_df = load_dataframe(load_adr, index_col="thresholds")
            acc_df = load_dataframe(load_acc, index_col="thresholds")
            far_df = load_dataframe(load_far, index_col="thresholds")
        except FileNotFoundError:
            print(f"files not found... constructing timeline")
            full_sweep(ncl=ncl, savedir=loaddir, hires=hires)

    else:
        tes_pop = TremorData().tes
        tes_pop.pop(3) # remove hard earthquake
        if hires:
            pp = construct_hires_timeline(ncl=ncl)
            look_forwards = np.round(np.linspace(0.5,7.5,endpoint=True,num=71),4)
        else:
            timeline = construct_timeline(ncl=ncl)
            pp = timeline.drop(['ys', 'full_calibrated', 'calibrated_prediction'], axis='columns')
            look_forwards = np.arange(1,7.5, step=0.5)

        thresholds = np.round(np.linspace(
            0.005, 0.05, num=96, endpoint=True), 4)

        alertday_ratios = dict()
        accuracies = dict()
        falsealert_ratios = dict()
        for lf in tqdm(look_forwards):
            # print(f"creating forecast model with lf = {lf}")
            fm = ForecastModel(ti='2011-01-01', tf='2020-01-01', window=2., overlap=0.75,
                            look_forward=lf, root=f'calibration_forecast_model', savefile_type='pkl')
            ys = pd.DataFrame(fm._get_label(pp.index.values), columns=['label'], index=pp.index)
            lf_alertday_ratios, lf_accuracies, lf_falsealert_ratios, _ = single_sweep(pp, ys, tes=tes_pop, lf=lf, thresholds=thresholds, inplace=True)
            alertday_ratios[lf] = lf_alertday_ratios
            accuracies[lf] = lf_accuracies
            falsealert_ratios[lf] = lf_falsealert_ratios
            # print(f"done")
        adr_df = pd.DataFrame(alertday_ratios,
                            index=[f'threshold_{th}'for th in thresholds]).add_prefix('lookforward_')
        adr_df.index.name = "thresholds"
        acc_df = pd.DataFrame(accuracies,
                            index=[f'threshold_{th}'for th in thresholds]).add_prefix('lookforward_')
        acc_df.index.name = "thresholds"
        far_df = pd.DataFrame(falsealert_ratios,
                            index=[f'threshold_{th}'for th in thresholds]).add_prefix('lookforward_')
        far_df.index.name = "thresholds"

        if savedir is None: savedir = f"{fm.rootdir}/calibration/contour"
        makedir(savedir)

        save_adr = f"{savedir}/alertdayratios_df__ncl_{ncl}.csv"
        save_acc = f"{savedir}/accuracies_df__ncl_{ncl}.csv"
        save_far = f"{savedir}/falsealertratios_windows_df__ncl_{ncl}.csv"

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


def plot_contours(ncl=100):
    ''' This function calls full_sweep() with saved dataframes then creates contour plot

    Try some more formatting using
    colorbar - https://stackoverflow.com/questions/15908371/matplotlib-colorbars-and-its-text-labels 
    contours - https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/contour_label_demo.html#sphx-glr-gallery-images-contours-and-fields-contour-label-demo-py
    '''
    fm = ForecastModel(ti='2011-01-01', tf='2020-01-01', window=2., overlap=0.75,
                       look_forward=2., root=f'calibration_forecast_model', savefile_type='pkl')
    tes_pop = TremorData().tes
    tes_pop.pop(3) # remove hard earthquake
    # load_adr = f"{fm.rootdir}/calibration/contour/alertdayratios_df.csv"
    # load_acc = f"{fm.rootdir}/calibration/contour/accuracies_df.csv"
    # load_far = f"{fm.rootdir}/calibration/contour/falsealertratios_df.csv"
    # load_far = f"{fm.rootdir}/calibration/contour/falsealertratios_windows_df.csv"
    # adr_df, acc_df, far_df = full_sweep(
    #     load_adr=load_adr, load_acc=load_acc, load_far=load_far)
    adr_df, acc_df, far_df = full_sweep(ncl=ncl) # Comment/Uncomment if need to create files
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

    fig, axs = plt.subplots(1,2,figsize=(10.5,18.5/3),sharey=True)
    col_names = acc_df.columns.values
    col_names = [float(x.split('_')[-1]) for x in col_names]
    row_names = acc_df.index.values
    row_names = [float(y.split('_')[-1]) for y in row_names]
    z = acc_df.values

    # Plot formatting
    for ax in axs:
        ax.grid(lw=0.5)
        ct = ax.contourf(col_names, row_names, z, colors=clist, levels=[i-.5 for i in range(6)], alpha=0.7)
        ax.tick_params(labelsize=8)
        ax.tick_params(axis='x', labelrotation=0.25)
        ax.set_xlabel('Lookforwards (days)', fontsize=12)
        ax.locator_params(axis='y', tight=True, nbins=10)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1,decimals=1))
    axs[0].set_ylabel('Probability Thresholds for alert', fontsize=12)

    adr_levels = np.linspace(0,0.4, num=10, endpoint=True)
    adr_vals = adr_df.values
    # far_levels = [1,0.985,0.98,0.975,0.95,0.945]
    far_levels = [0.96,0.95, 0.94,0.93,0.92]
    far_levels.sort()
    far_vals = far_df.values

    # Alert Day ratio contour lines
    axs[0].set_title('Alert Day Ratio', fontsize=16)
    adr_cs=axs[0].contour(col_names, row_names, adr_vals, levels=adr_levels, colors='black')
    # adr_cs=axs[0].contour(col_names, row_names, adr_vals, levels=adr_levels, cmap='Greys')
    axs[0].clabel(adr_cs, inline=True, fontsize=8)

    # False Alert ratio contour lines
    axs[1].set_title('False Alert Ratio', fontsize=16)
    far_cs = axs[1].contour(col_names, row_names, far_vals, levels=far_levels, colors='black')
    # far_cs = axs[1].contour(col_names, row_names, far_vals, levels=far_levels, cmap='Greys',vmin=0.5, vmax=1)
    axs[1].clabel(far_cs, inline=True, fontsize=8)

    # Pretty Colorbar formatting
    cb=fig.colorbar(ct)
    cb.ax.get_yaxis().set_ticks([])
    _, cb_xm, _, cb_ym = cb.ax.axis()
    for i in range(5):
        cb.ax.text(2, i, i, ha='center', va='center')
    cb.ax.get_yaxis().labelpad = 15
    cb.ax.set_ylabel('# of detected eruptions', rotation=270)
    # fig.set_size_inches(18.5, 10.5)
    # plt.show()
    save_plot=f"{fm.rootdir}/calibration/contour/contour_v6__ncl_{ncl}.png"
    plt.savefig(save_plot, format='png', dpi=300)
    plt.close()


def plot_hires_contours(ncl=100):
    ''' This function calls full_sweep() with saved dataframes then creates contour plot

    Try some more formatting using
    colorbar - https://stackoverflow.com/questions/15908371/matplotlib-colorbars-and-its-text-labels
    contours - https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/contour_label_demo.html#sphx-glr-gallery-images-contours-and-fields-contour-label-demo-py
    '''
    fm = ForecastModel(ti='2011-01-01', tf='2020-01-01', window=2., overlap=0.75,
                       look_forward=2., root=f'calibration_forecast_model', savefile_type='pkl')
    tes_pop = TremorData().tes
    tes_pop.pop(3) # remove hard earthquake
    loaddir = f"{fm.rootdir}/calibration/contour"
    adr_df, acc_df, far_df = full_sweep(loaddir=loaddir , ncl=ncl, hires=True)
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

    fig, axs = plt.subplots(1,2,figsize=(10.5,18.5/3),sharey=True)
    col_names = acc_df.columns.values
    col_names = [float(x.split('_')[-1]) for x in col_names]
    row_names = acc_df.index.values
    row_names = [float(y.split('_')[-1]) for y in row_names]
    z = acc_df.values

    # Plot formatting
    for ax in axs:
        ax.grid(lw=0.5)
        ct = ax.contourf(col_names, row_names, z, colors=clist, levels=[i-.5 for i in range(6)], alpha=0.7)
        ax.tick_params(labelsize=8)
        ax.tick_params(axis='x', labelrotation=0.25)
        ax.set_xlabel('Lookforwards (days)', fontsize=12)
        ax.locator_params(axis='y', tight=True, nbins=10)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1,decimals=1))
    axs[0].set_ylabel('Probability Thresholds for alert', fontsize=12)

    adr_levels = [0,0.08,0.09,0.1,0.11,0.12,0.2]
    adr_vals = adr_df.values
    far_levels = [0.97,0.94,0.92, 0.9,0.87]
    far_levels.sort()
    far_vals = far_df.values

    # Alert Day ratio contour lines
    axs[0].set_title('Alert Day Ratio', fontsize=16)
    adr_cs=axs[0].contour(col_names, row_names, adr_vals, levels=adr_levels, colors='black')
    # adr_cs=axs[0].contour(col_names, row_names, adr_vals, levels=adr_levels, cmap='Greys')
    # axs[0].clabel(adr_cs, inline=True, fontsize=8)
    fmt_adr = dict()
    for level in adr_levels:
        fmt_adr[level] = f'{level:.1%}'
    axs[0].clabel(adr_cs, inline=True, fontsize=8, fmt=fmt_adr)

    # False Alert ratio contour lines
    axs[1].set_title('False Alert Ratio', fontsize=16)
    far_cs = axs[1].contour(col_names, row_names, far_vals, levels=far_levels, colors='black')
    # far_cs = axs[1].contour(col_names, row_names, far_vals, levels=far_levels, cmap='Greys',vmin=0.5, vmax=1)
    # axs[1].clabel(far_cs, inline=True, fontsize=8)
    fmt_far = dict()
    for level in far_levels:
        fmt_far[level] = f'{level:.1%}'
    axs[1].clabel(far_cs, inline=True, fontsize=8, fmt=fmt_far)

    # Pretty Colorbar formatting
    cb=fig.colorbar(ct, ax=axs[0])
    cb.ax.get_yaxis().set_ticks([])
    for i in range(5):
        cb.ax.text(2, i, i, ha='center', va='center')
    cb.ax.get_yaxis().labelpad = 15
    cb.ax.set_ylabel('# of detected eruptions', rotation=270)
    save_plot=f"{fm.rootdir}/calibration/contour/hires_contour_test__ncl_{ncl}__v5.png"
    plt.savefig(save_plot, format='png', dpi=300)
    plt.close()


def decompose_to_weeks(filename, fnum_file):
    ''' Combined featfile has "fnum_{fnum_start}to{fnum_end}"

    Decompose featfile back into weeks for simplification of hires_forecasts()
    '''
    fnum_start = int(filename.split("fnum_")[-1].split('to')[0])
    fnum_end = int(filename.split("fnum_")[-1].split('to')[-1].split('.pkl')[0])

    df=load_dataframe(filename)
    fnums=load_dataframe(fnum_file)
    for fnum in fnums.index:
        if fnum <fnum_start: continue
        if fnum >fnum_end: break
        ti = fnums.iloc[fnum].ti
        tf = fnums.iloc[fnum].tf
        week = df.loc[(
            df.index >= ti) & (df.index <= tf)]
        f_save=f"{filename.split('_features__fnum_')[0]}__fnum_{fnum:03}_features.pkl"
        save_dataframe(week, f_save)
    test = load_dataframe(f_save)
    print(test.index)


def test_hires_calibration(plots=False, download_data=False):
    ''' This function creates weekly hires_forecasts using modeldir where te=[1,2,3,4,5,None]
    aka None has no holdout eruption.

    - Hires forecasts are performed weekly from '2011-01-01' to '2020-01-01'
    - Where the week does not land within month the None model is used
    - else the te model is used (precedence in the earlier eruption although could be possible to swap around or do averaging)

    The hires_forecasts will be saved somewhere and accessed later for plotting
    '''
    # set the available CPUs higher or lower as appropriate
    n_jobs = 3
    ncl=500

    offset = timedelta(minutes=10)

    def get_end_of_week(ti, tend):
        week = timedelta(weeks=1)
        if ti+week >= tend:
            return tend
        else:
            return ti+week

    tstart = datetimeify('2011-01-01')
    tend = datetimeify('2020-01-01')
    td = TremorData()
    if download_data:
        print("updating data")
        td.update(ti=tstart-timedelta(days=2), tf=tstart)
        print("done updating data")
    tis=[tstart,]
    check_ti = get_end_of_week(tis[-1], tend)
    while(check_ti != tend):
        tis.append(check_ti)
        check_ti = get_end_of_week(tis[-1], tend)

    tfs = list()
    for ti in tis[1:]:
        tfs.append(ti-offset)
    tfs.append(tend)
    # tfs = dict()
    # for ti in tis:
    #     t_endofweek = get_end_of_week(ti,tend)
    #     if t_endofweek == tend:
    #         tfs[ti] = tend
    #     else:
    #         tfs[ti] = t_endofweek-offset

    # Train and prepare all 6 models
    # eruption_nums = [0, 1, 2, 3, 4, None]
    eruption_nums = [None, 0, 1, 2, 3, 4]
    # fm_list = list()
    fm_dict = dict()
    for eruption_num in eruption_nums:
        fm_dict[eruption_num] = train_one_forecast_model(tstart=tstart, tend=tend, eruption_num=eruption_num, n_jobs=n_jobs, ncl=ncl)

    # Store ti and tf in csv
    # with open(f"{fm_list[0].rootdir}/calibration/tis_and_tfs.csv", "w") as f:
    with open(f"{fm_dict[0].rootdir}/calibration/tis_and_tfs.csv", "w") as f:
        f.write(f"ti,tf\n")
        for i, ti in enumerate(tis):
            f.write(f"{ti},{tfs[i]}\n")
        # for ti in tis:
        #     f.write(f"{ti},{tfs[ti]}\n")

    # Helper function to select forecast model from fm_list
    test_range = construct_exclude_dates()
    def which_eruption(time, test_times):
        '''takes time as datetime and list of test_times constructed using construct_exclude_dates()

        returns index value of eruption in eruption_nums
        NOTE: None is index -1
        '''
        # Biased towards earlier eruptions
        for i in range(5):
            if (test_times[i]['ti'] <= time) & (time < test_times[i]['tf']):
                return i
        return None
        # return -1

    # This is the looping through weeks to get hires forecasts
    for fnum, ti in enumerate(tis):
        if fnum < 226: continue # Skipping to  where fm got stuck
        # Loop through each fm and do feature extraction, i.e. call hires_forecast()
        for e_num in eruption_nums:
            iplot_name = f'{fm_dict[e_num].plotdir}/test_hires_forecast/interim_plots/hires_forecast__te_{e_num}__fnum_{fnum:03}__ti_{ti}__ncl_{ncl}.png'
            print(f"fnum={fnum:03}, e_num={e_num}, plot_name={iplot_name}")
            # forecast = fm_list[e_num].hires_forecast(ti=ti, tf=tfs[fnum], recalculate=True,
            #                   save=plot_name)

            hires_root = f"{fm_dict[e_num].root}_hires__fnum_{fnum:03}"
            forecast = fm_dict[e_num].hires_forecast(ti=ti, tf=tfs[fnum], recalculate=True,
                                                    root=hires_root, save=iplot_name)
            forecast = forecast.loc[(ti <= forecast.index) & (forecast.index <= tfs[fnum])]
            if_name = f'{fm_dict[e_num].rootdir}/calibration/interim_forecasts/te_{e_num}__fnum_{fnum:03}__ti_{ti}__ncl_{ncl}.pkl'
            save_dataframe(forecast, if_name, index_label='time')
            del forecast
        '''
        # Perform appending only if eruption is found (can easily call these once all hires_forecast() ie. feature extraction are performed
        e_num = which_eruption(ti, test_range)
        # if e_num == -1: # If no eruption within ti, check tf
        if e_num == None: # If no eruption within ti, check tf
            e_num = which_eruption(tfs[fnum], test_range)
        # plot_name = f'{fm_list[e_num].plotdir}/{fm_list[e_num].root}_hires_forecast__fnum_{fnum}__te_{eruption_nums[e_num]}.png'
        plot_name = f'{fm_dict[e_num].plotdir}/test_hires_forecast/hires_forecast__fnum_{fnum}__ti_{ti}__te_{e_num}.png'
        print(f"fnum={fnum}, e_num={e_num}, plot_name={plot_name}")
        # forecast = fm_list[e_num].hires_forecast(ti=ti, tf=tfs[fnum], recalculate=True,
        #                   save=plot_name)

        # Before each forecast download some data
        forecast = fm_dict[e_num].hires_forecast(ti=ti, tf=tfs[fnum], recalculate=True,
                          save=plot_name)
        if_name = f'{fm_dict[e_num].rootdir}/calibration/interim_forecasts/fnum_{fnum:03}__ti_{ti}__te_{e_num}.pkl'
        save_dataframe(forecast, if_name, index_label='time')
        if fnum == 0:
            hires_df = forecast
        else:
            forecast = forecast.loc[(ti <= forecast.index) & (forecast.index <= tfs[fnum])]
            f_name = f'{fm_dict[e_num].rootdir}/calibration/full_hires_consensus.pkl'
            hires_df = load_dataframe(f_name, index_col='time')
            hires_df = hires_df.append(forecast)
        # f_name = f'{fm_list[e_num].rootdir}/calibration/full_consensus.pkl'
        f_name = f'{fm_dict[e_num].rootdir}/calibration/full_hires_consensus.pkl'
        save_dataframe(hires_df, f_name, index_label='time')
        '''
        # memory management
        # del forecast
        # del hires_df
        # '''


    if plots:
        pass


def train_one_forecast_model(tstart, tend, eruption_num=None, n_jobs=3, retrain=False, ncl=100):
    month = timedelta(days=365.25 / 12)
    # construct forecast model object from trained modeldir
    data_streams = ['rsam', 'mf', 'hf', 'dsar']
    drop_features = ['linear_trend_timewise', 'agg_linear_trend']
    fm = ForecastModel(ti=tstart, tf=tend, window=2., overlap=0.75, n_jobs=n_jobs,
                       look_forward=2., data_streams=data_streams, root=f'calibration_forecast_model', savefile_type='pkl')
    # set modeldir but reuse features in root
    fm.modeldir = f'{fm.modeldir}__te_{eruption_num}__Ncl_{ncl}'
    if eruption_num is not None:
        exclude_range = construct_exclude_dates()[eruption_num]
        exclude_dates = [[exclude_range['ti'], exclude_range['tf']], ]
    else:
        exclude_dates = []
    fm.train(ti='2011-01-01', tf='2020-01-01', drop_features=drop_features, retrain=retrain,
             exclude_dates=exclude_dates, Ncl=ncl)

    return fm


def construct_exclude_dates():
    ''' Easily create exclude dates for each side of the eruption

    returns list of dicts
    '''
    # Initialise list of te and Tremor dates
    tes = TremorData().tes
    month = timedelta(days=365.25 / 12)
    exclude_dates = list()
    for te in tes:
        exclude_dates.append({'ti':te-month, 'tf':te+month})
    return exclude_dates

if __name__ == '__main__':
    # os.chdir('..')  # set working directory to root
    # calibration()
    # timeline_calibration()
    test_hires_calibration(download_data=False)
    # full_sweep()
    # plot_heatmap()
    # # plot_contours(ncl=500)
    # decompose_to_weeks(
    #     "/Users/teban/Documents/ADMIN/2020-21 Summer RA/PROGS/week7 - hires contour plots/calibration_forecast_model_hires_features__fnum_0to225.pkl",
    #     "/Users/teban/Documents/ADMIN/2020-21 Summer RA/sorenia_whakaari/calibration/tis_and_tfs.csv")
    plot_hires_contours(ncl=500)
