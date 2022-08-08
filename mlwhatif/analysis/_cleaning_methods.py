"""
Cleaning functions for the DataCleaning What-if Analysis
"""

import cleanlab
import numpy
import pandas
from numba import njit, prange
from pandas import DataFrame
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from mlwhatif.monkeypatching._monkey_patching_utils import wrap_in_mlinspect_array_if_necessary


def detect_outlier_standard_deviation(x, n_std=3.0, fitted_detector=None):
    """Standard Deviation Method (Univariate)"""
    # pylint: disable=invalid-name
    if fitted_detector is None:
        mean, std = numpy.mean(x), numpy.std(x)
        cut_off = std * n_std
        lower, upper = mean - cut_off, mean + cut_off
    else:
        lower, upper = fitted_detector
    return lambda y: (y > upper) | (y < lower), (lower, upper)


def detect_outlier_interquartile_range(x, k=1.5, fitted_detector=None):
    """Interquartile Range (Univariate)"""
    # pylint: disable=invalid-name
    if fitted_detector is None:
        q25, q75 = numpy.percentile(x, 25), numpy.percentile(x, 75)
        iqr = q75 - q25
        cut_off = iqr * k
        lower, upper = q25 - cut_off, q75 + cut_off
    else:
        lower, upper = fitted_detector
    return lambda y: (y > upper) | (y < lower), (lower, upper)


def detect_outlier_isolation_forest(x, contamination=0.01, fitted_detector=None):
    """Isolation Forest (Univariate)"""
    # pylint: disable=invalid-name
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
        """Outlier cleaning fit_transform for all different strategies"""
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
        annotation = (detection_strategy, fitted_detector, fitted_imputer)
        result_data._mlinspect_annotation = annotation  # pylint: disable=protected-access
        return result_data

    @staticmethod
    def transform_all(fit_data, input_df, column):
        """Outlier cleaning transform for all different strategies"""
        annotation = fit_data._mlinspect_annotation  # pylint: disable=protected-access
        detection_strategy, fitted_detector, fitted_imputer = annotation

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
        """Drop rows with missing values in that column"""
        return input_df.dropna(subset=[column])

    @staticmethod
    def fit_transform_all(input_df, column, strategy, cat=False):
        """MV cleaning fit_transform for all different strategies"""
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
            transformed_data._mlinspect_annotation = transformer  # pylint: disable=protected-access
        else:
            if cat is True:
                input_df = input_df.astype(str)
            transformer = SimpleImputer(strategy=strategy)
            input_df = transformer.fit_transform(input_df)
            transformed_data = wrap_in_mlinspect_array_if_necessary(input_df)
            transformed_data._mlinspect_annotation = transformer  # pylint: disable=protected-access
        return transformed_data

    @staticmethod
    def transform_all(fit_data, input_df, column, cat=False):
        """MV cleaning transform for all different strategies"""
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
    # pylint: disable=too-few-public-methods

    @staticmethod
    def drop_duplicates(input_df, column):
        """Drop rows containing duplicates in a selected column"""
        return input_df.drop_duplicates(subset=[column])


class MislabelCleaner:
    """
    Mislabel ErrorType
    """

    @staticmethod
    def fit_cleanlab(train_data, train_labels, make_classifier_func):
        """See https://github.com/cleanlab/cleanlab"""
        estimator = cleanlab.classification.CleanLearning(make_classifier_func())
        if isinstance(train_labels, pandas.Series):
            train_labels = train_labels.to_numpy()
        elif isinstance(train_labels, numpy.ndarray) and train_labels.ndim > 1:
            if train_labels.shape[1] != 1:
                raise Exception("TODO: Think about this edge case! "
                                "Cleanlab documentation: 'labels must be integers in 0, 1, …, K-1, where K is "
                                "the total number of classes'.")
            train_labels = train_labels.squeeze()
        estimator.fit(train_data, train_labels)
        return estimator

    @staticmethod
    def fit_shapley_cleaning(train_data, train_labels, make_classifier_func):
        """See https://arxiv.org/abs/2204.11131"""
        estimator = make_classifier_func()
        if isinstance(train_labels, pandas.Series):
            train_labels = train_labels.to_numpy()
        k = 10
        if k > (len(train_labels) * 0.2):
            train_data, test_data, train_labels, test_label = train_test_split(train_data, train_labels, test_size=k)
            shapley_values = MislabelCleaner._compute_shapley_values(train_data, train_labels, test_data, test_label, k)
            greater_zero = shapley_values >= 0.0
            train_data = train_data[greater_zero]
            train_labels = train_labels[greater_zero]
        estimator.fit(train_data, train_labels)
        return estimator

    # removed cache=True because of https://github.com/numba/numba/issues/4908 need a workaround soon
    @staticmethod
    @njit(fastmath=True, parallel=True)
    def _compute_shapley_values(X_train, y_train, X_test, y_test, K=1):
        # pylint: disable=invalid-name,too-many-locals
        """Compute approximate shapley values as presented in the DataScope paper. Here, we only do it for the
        estimator input data though and not for the input data of the surrounding pipeline.
        """
        # TODO: Without a clean test set, this is not guaranteed to help. We could also use the actual test set
        #  but that might be problematic from a data leakage perspective.
        #  We can think about this some more in the future.
        N = len(X_train)
        M = len(X_test)
        result = numpy.zeros(N, dtype=numpy.float32)

        for j in prange(M):  # pylint: disable=not-an-iterable
            score = numpy.zeros(N, dtype=numpy.float32)
            dist = numpy.zeros(N, dtype=numpy.float32)
            div_range = numpy.arange(1.0, N)
            div_min = numpy.minimum(div_range, K)
            for i in range(N):
                dist[i] = numpy.sqrt(numpy.sum(numpy.square(X_train[i] - X_test[j])))
            indices = numpy.argsort(dist)
            y_sorted = y_train[indices]
            eq_check = (y_sorted == y_test[j]) * 1.0
            diff = - 1 / K * (eq_check[1:] - eq_check[:-1])
            diff /= div_range
            diff *= div_min
            score[indices[:-1]] = diff
            score[indices[-1]] = eq_check[-1] / N
            score[indices] += numpy.sum(score[indices]) - numpy.cumsum(score[indices])
            result += score / M

        return result
