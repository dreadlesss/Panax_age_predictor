import os
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

__all__ = ['load_object', 'Data']


def load_object(fname):
    '''reload the object and results.

    Parameters
    ----------
    fname : str, default=None
        The name of the file.
        
    Returns
    -------
    results_obj : object
        The data object.
    '''
    pkl_file = open(fname, 'rb')
    results_obj = pickle.load(pkl_file)
    pkl_file.close()
    return results_obj


def standardize(scaler, x_fit=None, x_transform=None):
    '''Features standardization.
    
    Parameters
    ----------
    scaler : string {'MinMaxScaler', 'StandardScaler', 'Normalizer'}
        or callable {StandardScaler(), MinMaxScaler(), Normalizer()}
        The method that applied to the features.
        
    x_fit : array-like
        The data with at least one none zero component to train the scaler.
        
    x_transform : array-like
        The data to be transformed by scaler.
    
    Returns
    -------
    scaler : object
        The scaler object.
        
    x_output : array-like
        The transformed data.
    '''
    # initialize the scaler
    if isinstance(scaler, str):
        if scaler == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif scaler == 'StandardScaler':
            scaler = StandardScaler()
        elif scaler == 'Normalizer':
            scaler = Normalizer()

    # fit or transform
    if isinstance(scaler, (StandardScaler, MinMaxScaler, Normalizer)):
        if x_fit is not None:
            scaler.fit(x_fit)
        if x_transform is None:
            x_output = scaler.transform(x_fit)
        else:
            x_output = scaler.fit_transform(x_transform)
    return scaler, x_output


class Data():
    '''Load data and perform feature selection for the training and predicted data.

    Parameters
    ----------
    n_jobs : int, default=-1
        The number of cpu that will be used.
        
    data_path : str or path object
        The folder used to save the data.
    '''

    def __init__(self, n_jobs=-1, data_path='./data'):
        if n_jobs < 1:
            self.n_jobs = os.cpu_count()
        else:
            self.n_jobs = n_jobs
        self._train_data_exist = False
        self._predict_data_exist = False
        self._split = False
        self.data_path = Path(data_path)
        self.feature_expand = []
        self.feature_expand_params = {}

    def _valid_file(self, fname):
        '''Check if the file exists.
        
        Parameters
        ----------
        fname : str or path object
            The file that will be used.
            
        Returns
        -------
        file_path : path object
            The valid file path.
        '''
        path = Path(fname)
        file_path = Path('./data') / path.name
        if not file_path.exists():
            alter_path = Path(path.name)
            if alter_path.exists():
                shutil.move(alter_path, file_path)
            else:
                raise Exception('Can not find the %s' % fname)
        return file_path

    def _file_preprocess(self, file_path, drop_col=None):
        '''Read the file, drop the useless columns, return the data.
        
        Parameters
        ----------
        file_path : str
            File path contains the data.
            
        drop_col : str
            The useless columns that will be delete.
            
        Returns
        -------
        df : DataFrame
            The data (df) ready to use.
            
        path : path object
            The valid path of the data file.
        '''
        # load the file
        path = self._valid_file(file_path)
        if path.suffix == '.xlsx':
            df = pd.read_excel(path)
        elif path.suffix == '.tsv':
            df = pd.read_csv(path, delimiter='\t')
        elif path.suffix == '.csv':
            df = pd.read_csv(path, delimiter=',')

        # delete the usless columns
        if drop_col:
            df.drop(drop_col, axis=1, inplace=True)
        df.dropna(how='all', axis=0, inplace=True)
        df.dropna(how='all', axis=1, inplace=True)
        return df, path

    def load_data(self, file_path=None, label_col=None, drop_col=None, force=False, mode='train', df=None):
        '''Load data through a file, drop the useless columns, save the feature and label.

        Parameters
        ----------
        file_path : str, default=None
            File path contains the data.

        label_col : str, default=None
            Column name of the label.

        drop_col : list, default=None
            The name of useless columns.

        force : bool, default=False
            Reload the data by force.

        mode : str, default='train'
            Mode can be 'train' or 'predict'.
            
        df : DataFrame, default=None
            If it is provided, the file_path will be ignored.
            Otherwise, the data will be loaded through file_path.
        '''
        # prepare the data
        if df is None:
            df, file_path = self._file_preprocess(file_path, drop_col)

        # set the training data
        if mode == 'train':
            self.load_training_data(file_path, label_col, drop_col, df)
        # set the predict data
        elif mode == 'predict':
            self.load_predict_data(file_path, label_col, drop_col, df)
        else:
            raise Exception('Key error for the mode:%s.' % mode)

        # clear the record of feature expansion
        if force:
            self.feature_expand = []

    def load_training_data(self, file_path=None, label_col=None, drop_col=None, df=None):
        '''Load the training data.
        
        Parameters
        ----------
        Almost the same as self.load_data().
        '''
        # load file
        if df is None:
            df, file_path = self._file_preprocess(file_path)

        # set the predict data info
        self.file_path = file_path
        self.label_col = label_col

        # set the label, feature and columns
        self.df = df.copy(deep=True)
        self.df_y = df.pop(label_col)
        self.df_x = df.values
        self.df_x_columns = list(df.columns)
        self._train_data_exist = True
        print('Number of data set: %s' % (df.shape[0]))
        print('Number of features: %s' % (df.shape[1]))

    def load_predict_data(self, file_path=None, label_col=None, drop_col=None, df=None):
        '''Load the data used to predict.
        
        Parameters
        ----------
        Almost the same as self.load_data()
        '''
        # load file
        if df is None:
            df, file_path = self._file_preprocess(file_path)

        # set the training data info
        self.predict_file_path = file_path
        self.predict_label_col = label_col

        # set the label, feature and columns
        self.predict_df = df.copy(deep=True)
        self.predict_df_y = df.pop(label_col)
        self.predict_df_x = df.values
        self.predict_df_x_columns = list(df.columns)
        self._predict_data_exist = True
        print('Number of predicted data set: %s' % (df.shape[0]))
        print('Number of predicted features: %s' % (df.shape[1]))

    def feature_by_polynomial(self, degree=2, mode='train', **kwargs):
        '''Feature engineer: expand the features by PolynomicalFeatures.
        
        Parameters
        ----------
        degree : int, default=False
            The degree of polynomial features. Only degree = 2 is supported.
            
        mode : str, default='train'
            Mode can be 'train' or others.
            
        kwargs : 
            Other parameters that will be passed into the PolynomialFeatures funtion.
        '''
        if degree == 2:
            self.poly = PolynomialFeatures(degree=degree, include_bias=False, **kwargs)
            self.df_x = self.poly.fit_transform(self.df_x)
            # get the names of features after PolynomialFeatures
            columns = list(self.df_x_columns)
            df_x_columns = columns.copy()
            n = len(columns)
            for i in range(0, n):
                for j in range(i, n):
                    df_x_columns.append('{}*{}'.format(columns[i], columns[j]))
            if not (len(df_x_columns) == len(self.poly.get_feature_names())):
                raise Exception('PolynomialFeature transform error')
            self.df_x_columns = df_x_columns
        print('Number of existing features %s' % (self.df_x.shape[1]))

        # track and log the feature expansion procedure
        if mode == 'train':
            self.feature_expand.append(('feature_by_polynomial', self.poly))
            kwargs.update({'degree': degree, 'include_bias': False})
            self.feature_expand_params['feature_by_polynomial'] = kwargs

    def feature_by_variance(self, var_thresh=0, mode='train'):
        '''Feature engineer: delete features with near-zero variance.
        
        Parameters
        ----------
        var_thresh : float, default=0
            Features with a low variance will be deleted. 
            
        mode : str, default='train'
            Mode can be 'train' or others.
        '''
        selector = VarianceThreshold(var_thresh)
        try:
            var_selector_train = selector.fit_transform(self.df_x)
        except ValueError:
            print('No feature with near-zero variance.')
            self.var_selector_mask = None
        else:
            self.var_selector_mask = selector.get_support()
            self.df_x = var_selector_train
            self.df_x_columns = [i[0] for i in zip(self.df_x_columns, selector.get_support()) if i[1]]
        print('Number of existing features %s' % (self.df_x.shape[1]))

        # track and log the feature expansion procedure
        if mode == 'train':
            self.feature_expand.append(('feature_by_variance', self.var_selector_mask))
            self.feature_expand_params['feature_by_variance'] = {'var_thresh': var_thresh}

    def feature_by_correlation(self, high_corr=0, mode='train'):
        '''Feature engineer: delete features that have a high correlation with others.
        
        Parameters
        ----------
        high_corr : float, default=False
            Features with a high correlation will be deleted. 
            
        mode : str, default='train'
            Mode can be 'train' or others.
        '''
        # calculate the correlation coefficients
        np_corr = np.corrcoef(self.df_x, rowvar=False)
        np_corr_triu = np.abs(np.triu(np_corr, 1))
        np_corr_bool = np_corr_triu > high_corr
        df_corr_bool = pd.DataFrame(np_corr_bool, columns=self.df_x_columns, index=self.df_x_columns)
        df_data = pd.DataFrame(self.df_x, columns=self.df_x_columns)
        mask = np_corr_bool.any(axis=0)
        # the feature columns with corrcoef > threshold
        high_feat = [i[0] for i in zip(self.df_x_columns, mask) if i[1]]
        drop_set = set()
        for col in high_feat:
            # collect the feature pairs
            col_relates = df_corr_bool.index[df_corr_bool[col] == True]
            cal_cols = list(col_relates)
            cal_cols.append(col)
            # calculate the corrcoef of the feature and the label
            drop_dict = {}
            drop_dict = {cal_col: df_data[cal_col].corr(pd.Series(self.df_y)) for cal_col in cal_cols}
            # select the lower corrcoef and delete
            drop_set |= set([i[0] for i in sorted(drop_dict.items(), key=lambda d: (d[1], d[0]))[:-1]])
        support = [False if (i in drop_set) else True for i in self.df_x_columns]
        self.high_corr_mask = support
        self.df_x = self.df_x[:, support]
        self.df_x_columns = [i[0] for i in zip(self.df_x_columns, support) if i[1]]
        print('Number of existing features %s' % (self.df_x.shape[1]))

        # track and log the feature expansion procedure
        if mode == 'train':
            self.feature_expand.append(('feature_by_correlation', self.high_corr_mask))
            self.feature_expand_params['feature_by_correlation'] = {'high_corr': high_corr}

    def feature_by_k_best(self, k_best, score_func, mode='train'):
        '''Feature engineer: select the k best features.
        
        Parameters
        ----------
        k_best : int
            Select the k best features.  
            
        score_func : callable
            The critieria for select features.
            eg. chi2
        
        mode : str, default='train'
            Mode can be 'train' or others.
        '''
        selector = SelectKBest(score_func, k=k_best)
        self.df_x = selector.fit_transform(self.df_x, self.df_y)
        self.k_best_mask = selector.get_support()
        self.df_x_columns = [i[0] for i in zip(self.df_x_columns, self.k_best_mask) if i[1]]
        print('Number of existing features %s' % (self.df_x.shape[1]))

        # track and log the feature expansion procedure
        if mode == 'train':
            self.feature_expand.append(('feature_by_k_best', self.k_best_mask))
            self.feature_expand_params['feature_by_k_best'] = {'k_best': k_best, 'score_func': score_func}

    def feature_by_model(self, estimator, threshold, mode='train', **kwargs):
        '''Feature engineer: select features from models.
        
        Parameters
        ----------
        estimator : callable
            Model used to select features which has "feature_importances_" or "coef_" attribute.
            eg. LinearRegression
            
        threshold : str
            The threshold value to use for feature selection.
            e.g. "1.25*mean"
        
        mode : str, default='train'
            Mode can be 'train' or others.
            
        kwargs : 
            Other parameters that will be passed into the SelectFromModel funtion.
        '''
        selector = SelectFromModel(estimator=estimator, threshold=threshold, **kwargs)
        self.df_x = selector.fit_transform(self.df_x, self.df_y)
        self.model_mask = selector.get_support()
        self.df_x_columns = [i[0] for i in zip(self.df_x_columns, self.model_mask) if i[1]]
        print('Number of existing features %s' % (self.df_x.shape[1]))

        # track and log the feature expansion procedure
        if mode == 'train':
            self.feature_expand.append(('feature_by_model', self.model_mask))
            kwargs.update({'estimator': estimator, 'threshold': threshold})
            self.feature_expand_params['feature_by_model'] = kwargs

    def feature_by_RFECV(self, estimator, step=1, min_features=1, mode='train', **kwargs):
        '''Feature engineer: Select features by boruta.
        
        Parameters
        ----------
        estimator : callable estimator
            An estimator with "fit" method and provide importance either through
            a "coef_" attribute or "feature_importances_" attribute.

        step : int or float, default=1
            The number of features to remove at each iteration.
            eg. 5
            The percentage of features to remove at each iteration.
            eg. 0.5

        min_features : int, default=1
            The minimum number of features to be selected.
        
        mode : str, default='train'
            Mode can be 'train' or others.
            
        kwargs : 
            Other parameters that will be passed into the RFECV funtion.
        '''
        selector = RFECV(estimator, step=step, min_features_to_select=min_features)
        self.df_x = selector.fit_transform(self.df_x, self.df_y)
        self.rfe_mask = selector.get_support()
        self.df_x_columns = [i[0] for i in zip(self.df_x_columns, self.rfe_mask) if i[1]]
        print('Number of existing features %s' % (self.df_x.shape[1]))

        # track and log the feature expansion procedure
        if mode == 'train':
            self.feature_expand.append(('feature_by_RFECV', self.model_mask))
            kwargs.update({'estimator': estimator, 'step': step, 'min_features': min_features})
            self.feature_expand_params['feature_by_RFECV'] = kwargs

    def feature_standardize(self, scale):
        ''' Feature standardize and feature scaling.

        scale : {'MinMaxScaler', 'StandardScaler', 'Normalizer'}, default=None
            Specifies the scale method to be used in the feature scaling.
        '''
        self.scaler = None
        self.scaler, self.df_x = standardize(scale, self.df_x)
        self.feature_expand.append(('feature_standardize', self.scaler))
        self.feature_expand_params['feature_standardize'] = {'scaler': self.scaler}

    def data_sampling(self, stratify: ['stratify', 'random'] = 'stratify', test_size=0.2, rand=1):
        '''Split the dataset into training set and test set.
        If this function is executed, the data will be split into training and test set.
        Otherwise, the whole dataset will be used to training the model.
        
        Parameters
        ----------
        stratify : str, default='stratify'
            Stratify can be 'stratify' or 'random'. 'Stratify' means data is splited in a stratified fashion,
            'random' means the data is splited randomly.
        
        test_size : float, default=0.2
            The proportion of the test set.
            
        rand : int, default=1
            Random state to split the data.
        '''
        # stratified sampling
        self.stratify = stratify
        if stratify == 'stratify':
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.df_x, self.df_y,
                                                                                    test_size=test_size,
                                                                                    stratify=self.df_y,
                                                                                    random_state=rand)
        elif stratify == 'random':
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.df_x, self.df_y,
                                                                                    test_size=test_size, shuffle=True,
                                                                                    random_state=rand)
        else:
            raise Exception('stratify key error')

        self._split = True
        print('number of training set: %s' % (self.x_train.shape[0]))
        print('number of test set: %s' % (self.x_test.shape[0]))

    def engineer_for_prediction(self):
        '''Feature engineer for the predicted data.'''
        for func in self.feature_expand:
            item = func[0]
            mask = func[1]
            if item == 'feature_by_polynomial':
                # PolynomialFeatures
                self.predict_df_x = mask.transform(self.predict_df_x)

            elif item == 'feature_by_variance':
                # near-zero variance
                if mask is not None:
                    self.predict_df_x = self.predict_df_x[:, mask]

            elif item == 'feature_by_correlation':
                # delete features that have a high correlation with others
                if mask is not None:
                    self.predict_df_x = self.predict_df_x[:, mask]

            elif item == 'feature_by_k_best':
                # select the k best features
                if mask is not None:
                    self.predict_df_x = self.predict_df_x[:, mask]

            elif item == 'feature_by_model':
                # select the features from model
                if mask is not None:
                    self.predict_df_x = self.predict_df_x[:, mask]

            elif item == 'feature_by_RFECV':
                # select the features by recursive feature elimination
                if mask is not None:
                    self.predict_df_x = self.predict_df_x[:, mask]

            elif item == 'feature_standardize':
                # standardize
                if mask is not None:
                    self.predict_df_x = mask.transform(self.predict_df_x)
            print('Number of final features of predicted data %s' % (self.predict_df_x.shape[1]))

    def save_data(self, fname=None, ftime=True):
        '''Save the model object and results.

        Parameters
        ----------
        fname : str, default=None
            The file used for binary writing.

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