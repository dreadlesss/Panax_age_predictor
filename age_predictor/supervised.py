import pandas as pd
import numpy as np
import warnings
import pickle
import time
import os
from pathlib import Path
from copy import deepcopy
from functools import wraps

from IPython.core.display import display
from scipy.spatial import distance_matrix

from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBRegressor

from .feature_extractor import Data

__all__ = ['Model', 'RegressionModel']

MODELS = (
    {
        'LinearRegression': LinearRegression(),
        'Lasso': Lasso(),
        'Ridge': Ridge(),
        'ElasticNet': ElasticNet(),
        'KernelRidge': KernelRidge(),
        'BayesianRidge': BayesianRidge(),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'SVR_linear': SVR(),
        'SVR_rbf': SVR(),
        'SVR_poly': SVR(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'RandomForestRegressor': RandomForestRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
        'AdaBoostRegressor': AdaBoostRegressor(),
        'MLPRegressor': MLPRegressor(),
        'XGBRegressor': XGBRegressor()
    },
    {
        'LogisticRegression': LogisticRegression(),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'SVC_linear': SVC(),
        'SVC_rbf': SVC(),
        'SVC_poly': SVC(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'RandomForestClassifier': RandomForestClassifier(),
        'MLPClassifier': MLPClassifier(),
        'AdaBoostClassifier': AdaBoostClassifier(),
        'GaussianNB': GaussianNB(),
        'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
        'GradientBoostingClassifier': GradientBoostingClassifier()
    }
)


def _valid_candidates(candidates, mode):
    '''Validate the ML algorithms.
    
    Parameters
    ----------
    candidates : str
        The ML algorithms will be tuned.
        
    mode : str
        Mode can be 'regression' or non-regression.
        
    Returns
    -------
    valid_candidates : dict
        The valid candidates.
        
    _MODEL:
        The keys and functions of ML algorithms.
    '''
    # model used for regression and classification
    valid_candidates = deepcopy(candidates)
    if mode == 'regression':
        _MODEL = MODELS[0]
    else:
        _MODEL = MODELS[1]
    for item in candidates:
        if item not in _MODEL:
            del valid_candidates[item]
            warnings.warn('%s is not a valid candidate. Use supervised.MODELS to get valid options.' % item)
    return valid_candidates, _MODEL


def _applicability_domain(x_train=None, x_test=None, knn=5, weight=1.4, verbose=True):
    '''applicability domain (AD) of the model.
    The AD threshold is defined as follow:
    AD threshold = Dk + weight * std

    Parameters
    ----------
    x_train, x_test : array-like
        The training and test data used to compute the AD threshold.

    knn : int, default=5
        The k-nearest neighbors used to calculate the Dk.

    weight : float, default=1.4
        An empirical parameters that ranges from 1 - 5.
        
    verbose : bool, default=True
        If true, print the AD results.
        
    Returns
    -------
    reliable : list
        The results of AD.
        Reliable means the sample within the AD.
        Unreliable means the sample is out of the AD.
    '''
    # calc the distance matrix of training set
    train_dis = distance_matrix(x_train, x_train)
    train_knn_dis = np.sort(train_dis, axis=1)[:, 1:knn + 1]

    # threshold
    train_mean_dis = np.mean(train_knn_dis, axis=None)
    std = np.std(train_knn_dis, axis=None)
    threshold = train_mean_dis + weight * std

    # calc the distance matrix of test set
    test_dis = distance_matrix(x_test, x_train)
    test_knn_dis = np.sort(test_dis, axis=1)[:, :knn]

    # calc AD
    test_mean_dis = np.mean(test_knn_dis, axis=1)
    test_threshold = test_mean_dis

    reliable = ['Reliable' if i < threshold else 'Unreliable' for i in test_threshold]
    selected = np.where(test_threshold <= threshold)[0]
    if verbose:
        print('Total test data :', x_test.shape[0])
        print('Unreliable data :', reliable.count('Unreliable'))
        print('Reliable data :', reliable.count('Reliable'))
    return reliable


class Model():
    '''This class receives the cleaned data and build ML models automatically.

    Parameters
    ----------
    x_train, y_train : array-like
        The training data and the corresponding labels.

    x_test, y_test : array-like
        The test data and the corresponding labels.
        
    predict_x, predict_y : array-like
        The new data or the external test data and the corresponding labels.

    data : Data object
        Instantiated object from ./feature_extractor.

    max_iter : int, default=1000
        Parameters for the ML models.

    n_jobs : int, deafault=-1
        Number of cpu that will be used.

    data_path : str, default='./data'
        The directory used to save the results.
    '''

    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None, predict_x=None, predict_y=None, data=None,
                 max_iter=1000, n_jobs=-1, data_path='./data'):

        # set the training and test data
        self.load_training_data(x_train, y_train, x_test, y_test, data)

        # set the predicted data
        self.load_predict_data(predict_x, predict_y, data)

        # set the data path
        if isinstance(data, Data):
            self.data_path = data.data_path
        else:
            self.data_path = Path(data_path)

        if n_jobs < 1:
            self.n_jobs = os.cpu_count()
        else:
            self.n_jobs = n_jobs
        self.data_path = Path('./data')
        self.max_iter = max_iter
        self.predict_df_res_dict = {}
        self.AD_dict = {}

    def load_training_data(self, x_train=None, y_train=None, x_test=None, y_test=None, data=None):
        '''Load the training data.

        Parameters
        ----------
        x_train, y_train : array-like
            The training data and the corresponding labels.

        x_test, y_test : array-like
            The test data and the corresponding labels.

        data : Data object.
            Instantiated object from ./feature_extractor.
        '''
        if isinstance(data, Data):
            self.df_x = data.df_x
            self.df_y = data.df_y
            self._train_data_exist = True
            self.label_col = data.label_col
            self.df_x_columns = data.df_x_columns
            if data._split:
                self.x_train = data.x_train
                self.y_train = data.y_train
                self.x_test = data.x_test
                self.y_test = data.y_test
                self._split = True
                print('%s training data and %s test data detected in data!' % (
                    self.x_train.shape[0], self.x_test.shape[0]))
            else:
                self.x_train = data.df_x
                self.y_train = data.df_y
                self.x_test = data.df_x
                self.y_test = data.df_y
                self._split = False
                print('No test set detected in data! Whole data set will be used as a test set.')

        elif (x_train is not None) and (y_train is not None):
            self.x_train = x_train
            self.y_train = y_train
            self._train_data_exist = True
            if (x_test is not None) and (y_test is not None):
                self.x_test = x_test
                self.y_test = y_test
                self.df_x = np.concatenate([x_train, x_test])
                self.df_y = np.concatenate([y_train, y_test])
                self._split = True
                print('%s training data and %s test data received!' % (self.x_train.shape[0], self.x_test.shape[0]))
            else:
                self.x_test = x_train
                self.y_test = y_train
                self.df_x = x_train
                self.df_y = y_train
                self._split = False
                print('No test set received! Whole data set will be used as a test set.')
        #             self.df_x_columns = self.df_x.columns
        else:
            self._train_data_exist = False

    def load_predict_data(self, predict_x=None, predict_y=None, data=None):
        '''Load the predict data.

        Parameters
        ----------
        predict_x, predict_y : array-like
            The new data or the external test data and the corresponding labels.

        data : Data object.
            Instantiated object from ./feature_extractor.
        '''
        # set the predicted data
        if isinstance(data, Data):
            if data._predict_data_exist:
                self.predict_df = data.predict_df
                self.predict_df_y = data.predict_df_y
                self.predict_df_x = data.predict_df_x
                self.predict_label_col = data.predict_label_col
                self._predict_data_exist = True
            else:
                self._predict_data_exist = False
        elif predict_x is not None:
            self.predict_df_x = predict_x
            self.predict_df_y = predict_y
            self.predict_df = pd.concat([pd.DataFrame(predict_x), pd.DataFrame(predict_y)],
                                        columns=self.df_x_columns + ['label'])
            self.predict_label_col = 'label'
            self._predict_data_exist = True
        else:
            self._predict_data_exist = False

    def time_decorator(func):
        @wraps(func)
        def wrap(*args, **kwargs):
            time1 = time.time()
            res = func(*args, **kwargs)
            time2 = time.time()
            m, s = divmod((time2 - time1), 60)
            h, m = divmod(m, 60)
            print('{:=^70}\n'.format(' {} | Time: {}h {}min {:.0f}s '.format(func.__name__, h, m, s)))
            return res

        return wrap

    @time_decorator
    def model_tuning(self, scoring='neg_mean_squared_error', kfold_random_state=1, model_random_state=1):
        '''Tuning the parameters by GridSearchCV.
        
        Parameters
        ----------
        scoring : str or list, default='neg_mean_squared_error'
            Metrics to evaluate the model performance.
            
        kfold_random_state : int, default=1 
            Set the random_state for reproducibility of k-fold.
            
        model_random_state : int, default=1 
            Set the random_state for reproducibility of training model.
            
        Returns
        -------
        model_tuning_res : DataFrame
            All of the tuning results.
        '''
        self.scoring = scoring
        self.kfold_random_state = kfold_random_state
        self.model_random_state = model_random_state
        self.y_train = self.y_train.astype(float)
        self.y_test = self.y_test.astype(float)
        skf = KFold(n_splits=5, shuffle=True, random_state=kfold_random_state)
        self.KFold = skf
        candidates = self.candidates
        df_list = []
        for model_key in candidates:
            time1 = time.time()
            print('Tuning: ', model_key)
            estimator = self._MODEL[model_key]

            # set the params for models
            if hasattr(estimator, 'max_iter'):
                estimator.max_iter = self.max_iter
            if hasattr(estimator, 'n_jobs'):
                estimator.n_jobs = self.n_jobs
            if hasattr(estimator, 'random_state'):
                estimator.random_state = model_random_state
            model_params = candidates[model_key]

            # search the best parameters by GridSearchCV
            gsearch = GridSearchCV(estimator=estimator,
                                   param_grid=model_params,
                                   scoring=scoring,
                                   n_jobs=self.n_jobs,
                                   refit=False,
                                   cv=skf)
            gsearch.fit(self.x_train, self.y_train)
            df_model = pd.DataFrame(gsearch.cv_results_)
            df_model['algorithm'] = model_key
            df_list.append(df_model)
            time2 = time.time()
            print('Using time: %.2f\n' % (time2 - time1))

        self.model_tuning_res = pd.concat(df_list, ignore_index=True)
        self.model_tuning_res.drop(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time'], axis=1,
                                   inplace=True)
        return self.model_tuning_res

    @time_decorator
    def best_model_select(self, rank=None):
        '''Select the best parameters of each algorithm.
        
        Parameters
        ----------
        rank : str, default=None
            if two or more metrics were used in the tuning process,
            select one metrics used to rank the results.
            
        Returns
        -------
        best_df : DataFrame
            The best results selected by "rank".
        '''
        self.rank = rank
        res_df = self.model_tuning_res
        if isinstance(self.scoring, list):
            try:
                rank = 'rank_test_' + rank
                best_df = res_df[res_df[rank] == 1].copy(deep=True)
            except:
                raise Exception('select one of the metrics from %s' % self.scoring)
            else:
                self.display_cols = ['rank_test_' + i for i in self.scoring]
                self.display_cols.extend(['mean_test_' + i for i in self.scoring])
                self.display_cols.extend(['std_test_' + i for i in self.scoring])
        else:
            best_df = res_df[res_df['rank_test_score'] == 1].copy(deep=True)
            self.display_cols = ['mean_test_score', 'std_test_score', 'rank_test_score']
        self.best_df = best_df
        self.display_cols.extend(['algorithm'])
        self.display_cols.extend(['params'])
        display(best_df[self.display_cols])
        return best_df

    @time_decorator
    def best_model_evaluate(self, evaluate_metrics=['mean_squared_error'], with_coef=False, convert_acc=False):
        '''Add intercept, coefficient, metrics of the best model
        
        Parameters
        ----------
        evaluate_metrics : list, default=['mean_squared_error']
            The metrics to evaluate the model performance

        with_coef : bool, default=False
            Add the coefficient of the linear model.
            
        convert_acc : bool, default=False
            convert the continuous value to integer and calculate the accuracy of the model.
            
        Returns
        -------
        best_df : DataFrame
            The best models and their evaluation results.
        '''
        self.evaluate_metrics = evaluate_metrics
        column_list = ['intercept'] + self.df_x_columns
        coef_list = []
        index_list = []
        for index, row in self.best_df.iterrows():
            # set the model
            model_key = row['algorithm']
            model_params = row['params']
            estimator = self._MODEL[model_key]
            estimator.set_params(**model_params)

            # set the params
            if hasattr(estimator, 'max_iter'):
                estimator.max_iter = self.max_iter
            if hasattr(estimator, 'n_jobs'):
                estimator.n_jobs = self.n_jobs
            if hasattr(estimator, 'random_state'):
                estimator.random_state = self.model_random_state

            # train and predict
            estimator.fit(self.x_train, self.y_train)
            y_train_pred = estimator.predict(self.x_train)
            y_test_pred = estimator.predict(self.x_test)
            estimator.fit(self.df_x, self.df_y)
            y_pred = estimator.predict(self.df_x)

            # add the results into dataframe
            metric_cols = []
            eval_dict = {'train_': [y_train_pred, self.y_train],
                         'test_': [y_test_pred, self.y_test],
                         'all_': [y_pred, self.df_y]}
            for evaluate_metric in evaluate_metrics:
                for key, value in eval_dict.items():
                    # if dataset has not been splited
                    if (not self._split) and (key in ['train_', 'test_']):
                        continue
                    res = eval('metrics.%s' % evaluate_metric)(value[0], value[1])
                    self.best_df.loc[index, key + evaluate_metric] = res
                    metric_cols.extend([key + evaluate_metric])

                    # add the accuracy results
                    if convert_acc:
                        res = metrics.accuracy_score(self.number2class(value[0]), self.number2class(value[1]))
                        self.best_df.loc[index, key + 'acc'] = res
                        metric_cols.extend([key + 'acc'])
            # pickle the estimator
            try:
                self.best_df.loc[index, 'estimator'] = pickle.dumps(estimator)
            except:
                print('Warning! %s can not be pickled' % estimator)

            if with_coef:
                try:
                    coef = [estimator.intercept_]
                    coef += list(estimator.coef_)
                except:
                    coef = []
                coef_list.append(coef)
                index_list.append(index)

        # display
        self.display_cols.extend(metric_cols)
        display(self.best_df[self.display_cols])

        # add coef
        if with_coef:
            df_fom = pd.DataFrame(coef_list, columns=column_list, index=index_list)
            self.best_coef_res = pd.concat([df_fom, self.best_df], axis=1)
            return self.best_coef_res
        else:
            return self.best_df

    @time_decorator
    def best_model_predict(self, evaluate_metrics=['mean_squared_error'], tag='predict', convert_acc=False):
        '''Predict the unknown data using the best model and calc the metrics
        
        Parameters
        ----------
        evaluate_metrics : list, default=['mean_squared_error']
            The metrics to evaluate the model performance

        convert_acc : bool, default=False
            convert the continuous value to integer and calculate the accuracy of the model.
            
        Returns
        -------
        predict_df_res : DataFrame
            The predicted values of the best models.
        '''
        metrics_value_list = []
        y_pred_list = []
        y_index_list = []
        metric_cols = []
        flag = True

        for evaluate_metric in evaluate_metrics:
            # define suffix of the col name
            prefix = 'predict_' + evaluate_metric
            if prefix not in self.best_df.columns:
                suffix = ''
            else:
                col_name_metric = prefix
                n = 1
                while col_name_metric in self.best_df.columns:
                    col_name_metric = '_'.join([prefix, str(n)])
                    n += 1
                suffix = '_%s' % (n - 1)
            metric_name = 'predict_%s%s' % (evaluate_metric, suffix)

            for index, row in self.best_df.iterrows():
                est = pickle.loads(row['estimator'])
                y_valid_pred = est.predict(self.predict_df_x)
                if flag:
                    y_index_list.append(row['algorithm'])
                    y_pred_list.append(y_valid_pred)

                # performance evaluation
                try:
                    res = eval('metrics.%s' % evaluate_metric)(y_valid_pred, self.predict_df_y)
                except:
                    warnings.warn('Warning! %s not found in metrics' % evaluate_metric)
                else:
                    self.best_df.loc[index, metric_name] = res
                    if metric_name not in metric_cols:
                        metric_cols.append(metric_name)
                    # add the accuracy results
                    if convert_acc:
                        res = metrics.accuracy_score(self.number2class(y_valid_pred),
                                                     self.number2class(self.predict_df_y))
                        self.best_df.loc[index, 'predict_acc%s' % suffix] = res
                        metric_cols.extend(['predict_acc%s' % suffix])
            flag = False

        # combime the metrics results
        self.display_cols.extend(metric_cols)
        display(self.best_df[self.display_cols])
        df_res = pd.DataFrame(np.vstack(y_pred_list), index=y_index_list)
        self.predict_df_res = pd.concat([self.predict_df[[self.label_col]], df_res.T], axis=1)
        self.predict_df_res_dict[tag] = self.predict_df_res
        return self.predict_df_res

    def AD(self, knn=5, weight=1.4, verbose=True):
        '''An applicability domain is defined by knn and weight,
        which is used to determine how many samples within or out of AD.
        
        Parameters
        ----------
        knn : int, default=5
            The number of k-nearest neighbors.
            
        weight : float, default=1.4
            The weight of the standard deviation.
            
        verbose : bool
            Print the AD results
            
        Returns
        -------
        The number of samples within or out of AD.
        '''
        if len(self.predict_df_x) != self.predict_df_res.shape[0]:
            self.best_model_predict()
        ret = _applicability_domain(self.df_x, self.predict_df_x, knn, weight, verbose)
        self.predict_df_res['reliable'] = ret
        self.reliable = ret
        return ret.count('Unreliable'), ret.count('Reliable')

    def save_model(self, fname=None, ftime=True):
        '''save the object and results

        Parameters
        ----------
        fname : str, default=None
            The file name.

        time : bool, default=True
            Add the current time to the file name.
        '''
        if fname:
            fname = str(fname)
        if ftime:
            time_now = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
        else:
            time_now = ''

        name_comp = [i for i in [fname, time_now] if i]
        self.save_file = self.data_path / '{}.pkl'.format('_'.join(name_comp))

        output = open(self.save_file, 'wb')
        pickle.dump(self, output)
        output.close()


class RegressionModel(Model):
    def set_candidates(self, candidates):
        '''Set the algorithms and parameters for the regression models.
        
        Parameters
        ----------
        candidates : dict
            The algorithm keys and the parameters will be tuned.
            eg. {'Lasso':{'alpha': [0.01, 0.1, 1]}
        '''
        if not candidates:
            candidates = {
                'LinearRegression':
                    {},
                'Lasso':
                    {
                        'alpha': np.logspace(-7, 3, 400)
                    },
                'Ridge':
                    {
                        'alpha': np.logspace(-7, 3, 400)
                    },
                'ElasticNet':
                    {
                        'l1_ratio': np.arange(0, 1.05, 0.05),
                        'alpha': np.logspace(-7, 3, 200),
                        #                     'tol': np.logspace(-4, 1, 200)
                    },
                'KernelRidge':
                    {
                        'alpha': np.logspace(-4, -2, 100),
                        'gamma': np.logspace(-5, -3, 100),
                        'kernel': ["laplacian", "sigmoid"]
                    },
                'BayesianRidge':
                    {
                        'n_iter': np.arange(300, 600, 50),
                        'tol': np.arange(1e-5, 1e-4, 50),
                        'alpha_1': np.arange(1e-9, 1e-3, 50),
                        'alpha_2': np.arange(1e-9, 1e-3, 50),
                        'lambda_1': np.arange(1e-9, 1e-3, 50),
                        'lambda_2': np.arange(1e-9, 1e-3, 50)
                    },
                'KNeighborsRegressor':
                    {
                        'n_neighbors': np.arange(2, 30, 2),
                        'weights': ['uniform', 'distance'],
                        #                     'leaf_size': np.arange(20, 40, 5)
                    },
                'SVR_linear':
                    {
                        'kernel': ['linear'],
                        'C': np.logspace(-7, 3, 400),
                    },
                'SVR_rbf':
                    {
                        'kernel': ['rbf', 'sigmoid'],
                        'C': np.logspace(-7, 3, 30),
                        'gamma': np.logspace(-7, 3, 30),
                        'epsilon': np.logspace(-7, 3, 30)
                    },
                'SVR_poly':
                    {
                        'kernel': ['poly'],
                        'degree': [2],
                        'C': np.logspace(-7, 3, 30),
                        'gamma': np.logspace(-7, 3, 30),
                        'epsilon': np.logspace(-7, 3, 30)
                    },
                'DecisionTreeRegressor':
                    {
                        'max_depth': np.arange(5, 13, 1),
                        'min_samples_split': np.arange(2, 11, 2),
                        'min_samples_leaf': np.arange(2, 11, 2),
                        'max_features': ['auto', 'sqrt', 'log2']
                    },
                'RandomForestRegressor':
                    {
                        'n_estimators': np.arange(50, 500, 100),
                        'min_samples_split': np.arange(2, 12, 3),
                        'min_samples_leaf': np.arange(2, 12, 3),
                        'max_depth': np.arange(5, 12, 2),
                        'max_features': ['sqrt', 'log2', 'auto']
                    },
                'GradientBoostingRegressor':
                    {
                        'n_estimators': np.arange(50, 600, 50),
                        'learning_rate': [0.01, 0.1, 1],
                        'min_samples_split': np.arange(2, 11, 3),
                        'min_samples_leaf': np.arange(1, 11, 3),
                        'max_depth': np.arange(5, 12, 2),
                        'max_features': ['sqrt', 'log2', 'auto']
                    },
                'AdaBoostRegressor':
                    {
                        'n_estimators': np.arange(50, 500, 50),
                        'learning_rate': np.linspace(0.05, 0.5, 1),
                        'loss': ['linear', 'square', 'exponential']
                    },
                'MLPRegressor':
                    {
                        'hidden_layer_sizes': [(2, 2), (2, 4), (2, 6), (2, 8),
                                               (4, 2), (4, 4), (4, 6), (4, 8),
                                               (6, 2), (6, 4), (6, 6), (6, 8), ],
                        'activation': ['tanh'],
                        'learning_rate_init': [0.001, 0.01, 0.1],
                        'alpha': np.logspace(-7, 2, 10),
                        'early_stopping': [True],
                        'max_iter': [1000]
                    },
                'XGBRegressor':
                    {
                        'gamma': np.arange(0, 0.5, 0.1),
                        'max_depth': np.arange(3, 11),
                        'min_child_weight': np.arange(1, 20),
                        'colsample_bytree': np.arange(0.5, 1, 0.1),
                        'subsample': np.arange(0.5, 1, 0.1),
                        'learning_rate': np.arange(0.001, 0.2, 0.05),
                        'n_estimators': np.arange(10, 500, 100)
                    },
            }
        self.candidates, self._MODEL = _valid_candidates(candidates, 'regression')