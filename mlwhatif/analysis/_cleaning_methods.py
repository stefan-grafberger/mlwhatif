"""
Cleaning functions for the DataCleaning What-if Analysis
"""
import sys

import pandas
from pandas import DataFrame
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.impute import SimpleImputer

from mlwhatif.monkeypatching._monkey_patching_utils import wrap_in_mlinspect_array_if_necessary


class MissingValueCleaner:
    """
    Missing value ErrorType
    """
    @staticmethod
    def drop_missing(input_df, column):
        return input_df.dropna(subset=[column])

    @staticmethod
    def fit_transform_all(input_df, column, strategy):
        if strategy == 'mode':
            strategy = 'most_frequent'
        elif strategy == 'dummy':
            strategy = 'constant'
        if isinstance(input_df, DataFrame):
            transformer = SimpleImputer(strategy=strategy)
            if is_numeric_dtype(input_df[column]):
                input_df[column] = input_df[column].astype(str)
            input_df[[column]] = transformer.fit_transform(input_df[[column]])
            transformed_data = wrap_in_mlinspect_array_if_necessary(input_df)
            transformed_data._mlinspect_annotation = transformer
        else:
            transformer = SimpleImputer(strategy=strategy)
            input_df = transformer.fit_transform(input_df)
            transformed_data = wrap_in_mlinspect_array_if_necessary(input_df)
            transformed_data._mlinspect_annotation = transformer
        return transformed_data

    @staticmethod
    def transform_all(fit_data, input_df, column):
        transformer = fit_data._mlinspect_annotation  # pylint: disable=protected-access
        if isinstance(input_df, DataFrame):
            input_df[[column]] = transformer.transform(input_df[[column]])
        else:
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
