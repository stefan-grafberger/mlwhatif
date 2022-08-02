"""
Cleaning functions for the DataCleaning What-if Analysis
"""
import sys

import cleanlab
import numpy
import pandas
from pandas import DataFrame
from pandas.core.dtypes.common import is_numeric_dtype, is_bool_dtype
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from mlwhatif.monkeypatching._monkey_patching_utils import wrap_in_mlinspect_array_if_necessary


def detect_outlier_standard_deviation(x, n_std=3.0, fitted_detector=None):
    # Standard Deviation Method (Univariate)
    if fitted_detector is None:
        mean, std = numpy.mean(x), numpy.std(x)
        cut_off = std * n_std
        lower, upper = mean - cut_off, mean + cut_off
    else:
        lower, upper = fitted_detector
    return lambda y: (y > upper) | (y < lower), (lower, upper)


def detect_outlier_interquartile_range(x, k=1.5, fitted_detector=None):
    # Interquartile Range (Univariate)
    if fitted_detector is None:
        q25, q75 = numpy.percentile(x, 25), numpy.percentile(x, 75)
        iqr = q75 - q25
        cut_off = iqr * k
        lower, upper = q25 - cut_off, q75 + cut_off
    else:
        lower, upper = fitted_detector
    return lambda y: (y > upper) | (y < lower), (lower, upper)


def detect_outlier_isolation_forest(x, contamination=0.01, fitted_detector=None):
    # Isolation Forest (Univariate)
    if fitted_detector is None:
        isolation_forest = IsolationForest(contamination=contamination)
        if not isinstance(x, DataFrame):
            x = x.reshape(-1, 1)
        isolation_forest.fit(x)
        fitted_detector = isolation_forest
    result = fitted_detector.predict(x) == -1
    return result, fitted_detector


class OutlierCleaner:
    """
    Outlier ErrorType
    """

    @staticmethod
    def fit_transform_all(input_df, detection_strategy, repair_strategy, column):
        input_df = input_df.copy()
        data_to_repair = input_df[[column]]
        if detection_strategy == 'SD':
            outlier_indicator, fitted_detector = detect_outlier_standard_deviation(data_to_repair)
        elif detection_strategy == 'IQR':
            outlier_indicator, fitted_detector = detect_outlier_interquartile_range(data_to_repair)
        elif detection_strategy == 'IF':
            outlier_indicator, fitted_detector = detect_outlier_isolation_forest(data_to_repair)
        else:
            raise Exception(f"Unknown outlier detection strategy: {detection_strategy}!")

        data_to_repair[outlier_indicator] = numpy.nan
        fixed_data_with_fitted_imputer = MissingValueCleaner.fit_transform_all(data_to_repair, column, repair_strategy)
        fitted_imputer = fixed_data_with_fitted_imputer._mlinspect_annotation  # pylint: disable=protected-access
        input_df[[column]] = fixed_data_with_fitted_imputer

        result_data = wrap_in_mlinspect_array_if_necessary(input_df)
        result_data._mlinspect_annotation = (detection_strategy, fitted_detector, fitted_imputer)
        return result_data

    @staticmethod
    def transform_all(fit_data, input_df, column):
        detection_strategy, fitted_detector, fitted_imputer = \
            fit_data._mlinspect_annotation  # pylint: disable=protected-access
        if isinstance(input_df, DataFrame):
            data_to_repair = input_df[[column]]
        else:
            data_to_repair = input_df
        if detection_strategy == 'SD':
            outlier_indicator, fitted_detector = detect_outlier_standard_deviation(data_to_repair,
                                                                                   fitted_detector=fitted_detector)
        elif detection_strategy == 'IQR':
            outlier_indicator, fitted_detector = detect_outlier_interquartile_range(data_to_repair,
                                                                                    fitted_detector=fitted_detector)
        elif detection_strategy == 'IF':
            outlier_indicator, fitted_detector = detect_outlier_isolation_forest(data_to_repair,
                                                                                 fitted_detector=fitted_detector)
        else:
            raise Exception(f"Unknown outlier detection strategy: {detection_strategy}!")
        data_to_repair[outlier_indicator] = numpy.nan
        data_to_repair = fitted_imputer.transform(data_to_repair)
        if isinstance(input_df, DataFrame):
            input_df[[column]] = data_to_repair
        else:
            input_df = data_to_repair
        return input_df


class MissingValueCleaner:
    """
    Missing value ErrorType
    """
    @staticmethod
    def drop_missing(input_df, column):
        return input_df.dropna(subset=[column])

    @staticmethod
    def fit_transform_all(input_df, column, strategy, cat=False):
        input_df = input_df.copy()
        if strategy == 'mode':
            strategy = 'most_frequent'
        elif strategy == 'dummy':
            strategy = 'constant'
        if isinstance(input_df, DataFrame):
            transformer = SimpleImputer(strategy=strategy)
            if cat is True:
                input_df[column] = input_df[column].astype(str)
            input_df[[column]] = transformer.fit_transform(input_df[[column]])
            transformed_data = wrap_in_mlinspect_array_if_necessary(input_df)
            transformed_data._mlinspect_annotation = transformer
        else:
            if cat is True:
                input_df = input_df.astype(str)
            transformer = SimpleImputer(strategy=strategy)
            input_df = transformer.fit_transform(input_df)
            transformed_data = wrap_in_mlinspect_array_if_necessary(input_df)
            transformed_data._mlinspect_annotation = transformer
        return transformed_data

    @staticmethod
    def transform_all(fit_data, input_df, column, cat=False):
        transformer = fit_data._mlinspect_annotation  # pylint: disable=protected-access
        if isinstance(input_df, DataFrame):
            if cat is True:
                input_df[column] = input_df[column].astype(str)
            input_df[[column]] = transformer.transform(input_df[[column]])
        else:
            if cat is True:
                input_df = input_df.astype(str)
            input_df = transformer.transform(input_df)
        return input_df


class DuplicateCleaner:
    """
    Missing value ErrorType
    """
    @staticmethod
    def drop_duplicates(input_df, column):
        return input_df.drop_duplicates(subset=[column])


class MVCleaner(object):
    # pylint: disable-all
    def __init__(self, method='delete', **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.is_fit = False
        if method == 'impute':
            if 'num' not in kwargs or 'cat' not in kwargs:
                print("Must give imputation method for numerical and categorical data")
                sys.exit(1)
            self.tag = "impute_{}_{}".format(kwargs['num'], kwargs['cat'])
        else:
            self.tag = "delete"

    def detect(self, df):
        return df.isnull()

    def fit(self, dataset, df):
        if self.method == 'impute':
            num_method = self.kwargs['num']
            cat_method = self.kwargs['cat']
            num_df = df.select_dtypes(include='number')
            cat_df = df.select_dtypes(exclude='number')
            if num_method == "mean":
                num_imp = num_df.mean()
            if num_method == "median":
                num_imp = num_df.median()
            if num_method == "mode":
                num_imp = num_df.mode().iloc[0]

            if cat_method == "mode":
                cat_imp = cat_df.mode().iloc[0]
            if cat_method == "dummy":
                cat_imp = ['missing'] * len(cat_df.columns)
                cat_imp = pandas.Series(cat_imp, index=cat_df.columns)
            self.impute = pandas.concat([num_imp, cat_imp], axis=0)
        self.is_fit = True

    def repair(self, df):
        if self.method == 'delete':
            df_clean = df.dropna()

        if self.method == 'impute':
            df_clean = df.fillna(value=self.impute)
        return df_clean

    def clean_df(self, df):
        if not self.is_fit:
            print('Must fit before clean.')
            sys.exit()
        mv_mat = self.detect(df)
        df_clean = self.repair(df)
        return df_clean, mv_mat


class MislabelCleaner:
    """
    Mislabel ErrorType
    """

    @staticmethod
    def fit_cleanlab(train_data, train_labels, make_classifier_func):
        estimator = cleanlab.classification.CleanLearning(make_classifier_func())
        if isinstance(train_labels, pandas.Series):
            train_labels = train_labels.to_numpy()
        estimator.fit(train_data, train_labels)
        return estimator
