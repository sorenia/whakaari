import joblib
import os, sys ### --- added sys
sys.path.insert(0, os.path.abspath('..')) ### --- added line
# os.chdir('..')  # set working directory to root
from glob import glob

import matplotlib.pyplot as plt
from whakaari import TremorData, ForecastModel, save_dataframe, load_dataframe, datetimeify
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
        "calibrated_prediction": calibrated_classifier.predict_proba(X_full)[:, 1],
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

    # # Generate and store forecast model outputs - Calibrators are there for later comparisons
    # calibrators = list()
    # for i in range(5):
    #     calibrator = calibration(eruption_num=i, plots=False)
    #     f_name = f"{fm.rootdir}/calibration/calibration_forecast_model__te_{i}.pkl"
    #     save_dataframe(calibrator['predictions'], f_name, index_label = 'time')
    #     calibrators.append(calibrator)
    # calibrator = calibration(eruption_num=None, plots=False)
    # f_name = f"{fm.rootdir}/calibration/calibration_forecast_model__te_None.pkl"
    # save_dataframe(calibrator['predictions'], f_name, index_label = 'time')
    # calibrators.append(calibrator)

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

    fm_5 = ForecastModel(ti='2011-01-01', tf='2020-01-01', window=2., overlap=0.75,
                       look_forward=5., data_streams=data_streams, root=f'calibration_forecast_model', savefile_type='pkl')
    # run sigmoid calibration
    ys = pd.DataFrame(fm_5._get_label(timeline.index.values), columns=['label'], index=timeline.index)
    a,b = _sigmoid_calibration(timeline.prediction, ys)
    with open(f"{fm.rootdir}/calibration/sigmoid_parameters__full_calibrated.csv", "w") as f:
        f.write(f"a,b\n{a},{b}")

    # and apply function
    def get_calibrated(prediction):
        return expit(-(a * prediction + b))
    timeline['full_calibrated'] = timeline['prediction'].apply(get_calibrated)
    timeline['ys'] = ys

    f_save = f"{fm.rootdir}/calibration/calibration_forecast_model__TIMELINE.pkl"
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
