# pylint: disable-all
import datetime
import os
import random
import sys

import numpy as np
import pandas as pd
from faker import Faker
from jenga.corruptions.generic import CategoricalShift
from jenga.corruptions.numerical import GaussianNoise, Scaling
from jenga.corruptions.text import BrokenCharacters
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, RobustScaler
from sklearn.preprocessing import label_binarize

from experiments.manual_analysis.cleaner_copy_from_non_pip_cleanml import MVCleaner, OutlierCleaner, DuplicatesCleaner
from mlwhatif.utils import get_project_root

data_root = os.path.join(str(get_project_root()), "experiments", "end_to_end", "../end_to_end/datasets")


def get_dataset():
    def load_data():
        reviews = pd.read_csv(os.path.join(data_root, "reviews", "reviews.csv.gz"), compression='gzip',
                              index_col=0)
        ratings = pd.read_csv(os.path.join(data_root, "reviews", "ratings.csv"), index_col=0)
        products = pd.read_csv(os.path.join(data_root, "reviews", "products.csv"), index_col=0)
        categories = pd.read_csv(os.path.join(data_root, "reviews", "categories.csv"), index_col=0)

        return reviews, ratings, products, categories

    def integrate_data(reviews, ratings, products, categories):
        start_date = datetime.date(year=2011, month=6, day=22)

        reviews = reviews[reviews['review_date'] >= start_date.strftime('%Y-%m-%d')]

        reviews_with_ratings = reviews.merge(ratings, on='review_id')
        products_with_categories = products.merge(left_on='category_id', right_on='id', right=categories)

        random_categories = ['Digital_Video_Games']
        products_with_categories = products_with_categories[
            products_with_categories['category'].isin(random_categories)]

        reviews_with_products_and_ratings = reviews_with_ratings.merge(products_with_categories, on='product_id')

        return reviews_with_products_and_ratings

    def compute_feature_and_label_data(reviews_with_products_and_ratings, final_columns):
        reviews_with_products_and_ratings['is_helpful'] = reviews_with_products_and_ratings['helpful_votes'] > 0

        projected_reviews = reviews_with_products_and_ratings[final_columns]

        split_date = datetime.date(year=2013, month=12, day=20)

        train_data = projected_reviews[projected_reviews['review_date'] <= split_date.strftime('%Y-%m-%d')]
        train_labels = label_binarize(train_data['is_helpful'], classes=[True, False]).ravel()

        test_data = projected_reviews[projected_reviews['review_date'] > split_date.strftime('%Y-%m-%d')]
        test_labels = label_binarize(test_data['is_helpful'], classes=[True, False]).ravel()

        return train_data, train_labels, test_data, test_labels

    numerical_columns = ['total_votes', 'star_rating']
    categorical_columns = ['vine', 'category']
    text_columns = ["review_body"]
    # final_columns = numerical_columns + categorical_columns + text_columns + ['is_helpful', 'review_date']
    final_columns = numerical_columns + categorical_columns + text_columns + ['is_helpful', 'review_date', "review_id"]

    reviews, ratings, products, categories = load_data()

    integrated_data = integrate_data(reviews, ratings, products, categories)
    train, train_labels, test, test_labels = compute_feature_and_label_data(integrated_data, final_columns)

    return train, train_labels, test, test_labels, numerical_columns, categorical_columns, text_columns


def apply_clean_ml_for_variant(test, test_labels, train, train_labels, variant_index):
    """Apply CleanML data cleaning"""
    if variant_index < 3:  # column
        column = 'category'
        if variant_index == 0:
            cleaner = MVCleaner("delete")
        elif variant_index == 1:
            cleaner = MVCleaner("impute", num="mean", cat="mode")
        elif variant_index == 2:
            cleaner = MVCleaner("impute", num="mean", cat="dummy")
    elif variant_index < 6:  # variant
        column = 'vine'
        if variant_index == 3:
            cleaner = MVCleaner("delete")
        elif variant_index == 4:
            cleaner = MVCleaner("impute", num="mean", cat="mode")
        elif variant_index == 5:
            cleaner = MVCleaner("impute", num="mean", cat="dummy")
    elif variant_index < 10:  # variant
        column = 'star_rating'
        if variant_index == 6:
            cleaner = MVCleaner("delete")
        elif variant_index == 7:
            cleaner = MVCleaner("impute", num="median", cat="mode")
        elif variant_index == 8:
            cleaner = MVCleaner("impute", num="mean", cat="dummy")
        elif variant_index == 9:
            cleaner = MVCleaner("impute", num="mode", cat="dummy")
    elif variant_index < 19:  # variant
        column = 'total_votes'
        if variant_index == 10:
            cleaner = OutlierCleaner(detect_method="SD", repairer=MVCleaner("impute", num="mean", cat="dummy"))
        elif variant_index == 11:
            cleaner = OutlierCleaner(detect_method="SD", repairer=MVCleaner("impute", num="mode", cat="dummy"))
        elif variant_index == 12:
            cleaner = OutlierCleaner(detect_method="SD", repairer=MVCleaner("impute", num="median", cat="dummy"))
        elif variant_index == 13:
            cleaner = OutlierCleaner(detect_method="IQR", repairer=MVCleaner("impute", num="mean", cat="dummy"))
        elif variant_index == 14:
            cleaner = OutlierCleaner(detect_method="IQR", repairer=MVCleaner("impute", num="mode", cat="dummy"))
        elif variant_index == 15:
            cleaner = OutlierCleaner(detect_method="IQR", repairer=MVCleaner("impute", num="median", cat="dummy"))
        elif variant_index == 16:
            cleaner = OutlierCleaner(detect_method="IF", repairer=MVCleaner("impute", num="mean", cat="dummy"))
        elif variant_index == 17:
            cleaner = OutlierCleaner(detect_method="IF", repairer=MVCleaner("impute", num="mode", cat="dummy"))
        elif variant_index == 18:
            cleaner = OutlierCleaner(detect_method="IF", repairer=MVCleaner("impute", num="median", cat="dummy"))
    elif variant_index < 20:  # variant
        column = 'review_id'
        if variant_index == 19:
            cleaner = DuplicatesCleaner()
    if variant_index != -1:
        train_input = train[[column]]
        test_input = test[[column]]
        cleaner.fit(None, train_input)

        if variant_index in {0, 3, 6, 19}:  # Filter cleaners
            train_mask = ~cleaner.detect(train_input).iloc[:, 0].to_numpy()
            test_mask = ~cleaner.detect(test_input).iloc[:, 0].to_numpy()

            train = train[train_mask]
            test = test[test_mask]
            train_labels = train_labels[train_mask]
            test_labels = test_labels[test_mask]
        else:  # Projection cleaners
            train[[column]], _ = cleaner.clean_df(train_input)
            test[[column]], _ = cleaner.clean_df(test_input)
    return test, test_labels, train, train_labels


def apply_jenga_for_variant(test, variant_index):
    """Apply Jenga data corruptions"""
    fraction = [0.2, 0.4, 0.6, 0.8, 1.0][variant_index % 5]
    if variant_index < 5:
        corruption = BrokenCharacters(column='review_body', fraction=fraction)
    elif variant_index < 10:
        corruption = CategoricalShift(column='vine', fraction=fraction)
    elif variant_index < 15:
        corruption = CategoricalShift(column='category', fraction=fraction)
    elif variant_index < 20:
        corruption = Scaling(column='total_votes', fraction=fraction)
    elif variant_index < 25:
        corruption = GaussianNoise(column='star_rating', fraction=fraction)
    else:
        raise ValueError(f"Invalid variant_index: variant_index!")
    if variant_index != -1:
        test = corruption.transform(test)
    return test


def get_featurization(numerical_columns, categorical_columns, text_columns):
    transformers = [('num', RobustScaler(), numerical_columns)]
    if len(text_columns) >= 1:
        assert len(text_columns) == 1
        transformers.append(('text', HashingVectorizer(n_features=2 ** 5), text_columns[0]))
    for cat_column in categorical_columns:
        def another_imputer(df_with_categorical_columns):
            return df_with_categorical_columns.fillna('__missing__').astype(str)

        cat_pipe = Pipeline([('anothersimpleimputer', FunctionTransformer(another_imputer)),
                             ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))])

        transformers.append((f"cat_{cat_column}", cat_pipe, [cat_column]))

    featurization = ColumnTransformer(transformers)
    return featurization


def get_model():
    model = SGDClassifier(loss='log', max_iter=30, n_jobs=1)
    return model


def main_function():
    analysis = 'none'
    # analysis = 'data_corruption'
    # analysis = 'data_cleaning'
    if len(sys.argv) > 1:
        analysis = sys.argv[1]
    if analysis == 'data_corruption':
        variant_count = 25
    elif analysis == 'data_cleaning':
        variant_count = 20
    elif analysis == 'none':
        variant_count = 0
    else:
        raise ValueError(f"Invalid analysis!")
    seed = 42
    if len(sys.argv) > 2:
        seed = sys.argv[2]
    np.random.seed(seed)
    random.seed(seed)
    scores = []
    train, train_labels, test, test_labels, numerical_columns, categorical_columns, text_columns = \
        get_dataset()
    featurization = get_featurization(numerical_columns, categorical_columns, text_columns)
    model = get_model()
    pipeline = Pipeline([('featurization', featurization),
                         ('model', model)])
    pipeline = pipeline.fit(train, train_labels)
    for variant_index in range(-1, variant_count):
        print(f'Running fast_loading featurization featurization_1 on dataset reviews with model '
              f'logistic_regression with analysis {analysis} for variant {variant_index}')
        if analysis == "data_corruption":
            test = apply_jenga_for_variant(test, variant_index)
        elif analysis == "data_cleaning":
            raise ValueError("Train reuse is only possible for corruption!")
        predictions = pipeline.predict(test)
        score = accuracy_score(test_labels, predictions)
        print('    Score: ', score)
        scores.append(score)
    results = pd.DataFrame({'variant_index': range(-1, variant_count), 'score': scores})
    print(results)


# Make sure this code is not executed during imports
if sys.argv[0] == "manual_robustness" or __name__ == "__main__":
    main_function()
