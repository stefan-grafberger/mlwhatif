import datetime
import os
import random
import re
import string
import sys

import numpy as np
import pandas as pd
from faker import Faker
from jenga.corruptions.generic import CategoricalShift
from jenga.corruptions.numerical import GaussianNoise, Scaling
from jenga.corruptions.text import BrokenCharacters
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

from experiments.manual_analysis.cleaner_copy_from_non_pip_cleanml import MVCleaner
from mlwhatif.utils import get_project_root

data_root = os.path.join(str(get_project_root()), "experiments", "end_to_end", "../end_to_end/datasets")


def get_dataset(seed, analysis_name, variant_index):
    def random_subset(arr):
        size = np.random.randint(low=1, high=len(arr) + 1)
        choice = np.random.choice(arr, size=size, replace=False)
        return [str(item) for item in choice]

    def load_data():
        reviews = pd.read_csv(os.path.join(data_root, "reviews", "reviews.csv.gz"), compression='gzip',
                              index_col=0)
        ratings = pd.read_csv(os.path.join(data_root, "reviews", "ratings.csv"), index_col=0)
        products = pd.read_csv(os.path.join(data_root, "reviews", "products.csv"), index_col=0)
        categories = pd.read_csv(os.path.join(data_root, "reviews", "categories.csv"), index_col=0)

        return reviews, ratings, products, categories

    def integrate_data(reviews, ratings, products, categories, fake):
        start_date = fake.date_between(start_date=datetime.date(year=2011, month=1, day=1),
                                       end_date=datetime.date(year=2013, month=6, day=1))

        reviews = reviews[reviews['review_date'] >= start_date.strftime('%Y-%m-%d')]

        reviews_with_ratings = reviews.merge(ratings, on='review_id')
        products_with_categories = products.merge(left_on='category_id', right_on='id', right=categories)

        random_categories = random_subset(list(categories.category))
        products_with_categories = products_with_categories[
            products_with_categories['category'].isin(random_categories)]

        reviews_with_products_and_ratings = reviews_with_ratings.merge(products_with_categories, on='product_id')

        return reviews_with_products_and_ratings

    def compute_feature_and_label_data(reviews_with_products_and_ratings, final_columns, fake):
        reviews_with_products_and_ratings['is_helpful'] = reviews_with_products_and_ratings['helpful_votes'] > 0

        projected_reviews = reviews_with_products_and_ratings[final_columns]

        split_date = fake.date_between(start_date=datetime.date(year=2013, month=12, day=1),
                                       end_date=datetime.date(year=2015, month=1, day=1))

        train_data = projected_reviews[projected_reviews['review_date'] <= split_date.strftime('%Y-%m-%d')]
        train_labels = label_binarize(train_data['is_helpful'], classes=[True, False]).ravel()

        test_data = projected_reviews[projected_reviews['review_date'] > split_date.strftime('%Y-%m-%d')]
        test_labels = label_binarize(test_data['is_helpful'], classes=[True, False]).ravel()

        return train_data, train_labels, test_data, test_labels

    numerical_columns = ['total_votes', 'star_rating']
    categorical_columns = ['vine', 'category']
    text_columns = ["review_body"]
    final_columns = numerical_columns + categorical_columns + text_columns + ['is_helpful', 'review_date']

    reviews, ratings, products, categories = load_data()

    fake = Faker()
    fake.seed_instance(seed)
    integrated_data = integrate_data(reviews, ratings, products, categories, fake)
    train, train_labels, test, test_labels = compute_feature_and_label_data(integrated_data, final_columns, fake)

    if analysis_name == "data_corruption":
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
        test = corruption.transform(test)
    elif analysis_name == "data_cleaning":
        if variant_index <= 3:  # column
            if variant_index <= 3:  # variant
                column = 'category'
                if variant_index == 0:
                    cleaner = MVCleaner("delete")
                if variant_index == 1:
                    cleaner = MVCleaner("impute", num="mean", cat="mode")
                if variant_index == 2:
                    cleaner = MVCleaner("impute", num="mean", cat="dummy")

        # FIXME: This does not work for dropna yet, we need to use the indicator array!
        train_input = train[[column]]
        test_input = test[[column]]
        cleaner.fit(None, train_input)

        if variant_index in {0}:  # Filter cleaners
            train_mask = ~cleaner.detect(train_input).iloc[:, 0].to_numpy()
            test_mask = ~cleaner.detect(test_input).iloc[:, 0].to_numpy()

            train = train[train_mask]
            test = test[test_mask]
            train_labels = train_labels[train_mask]
            test_labels = test_labels[test_mask]
        else:  # Projection cleaners
            train[[column]] = cleaner.repair(train_input)
            test[[column]] = cleaner.repair(test_input)

    return train, train_labels, test, test_labels, numerical_columns, categorical_columns, text_columns


def get_featurization(numerical_columns, categorical_columns, text_columns):
    # based on dspipes 'num_pipe_3' and 'txt_pipe_2'
    union = FeatureUnion([("pca", PCA(n_components=(len(numerical_columns) - 1))),
                          ("svd", TruncatedSVD(n_iter=1, n_components=(len(numerical_columns) - 1)))])
    num_pipe = Pipeline([('union', union), ('scaler', StandardScaler())])

    transformers = [('num', num_pipe, numerical_columns)]
    if len(text_columns) >= 1:
        assert len(text_columns) == 1

        def remove_numbers(text_array):
            return [str(re.sub(r'\d+', '', text)) for text in text_array]

        def remove_punctuation(text_array):
            translator = str.maketrans('', '', string.punctuation)
            return [text.translate(translator) for text in text_array]

        def text_lowercase(text_array):
            return list(map(lambda x: x.lower(), text_array))

        def remove_urls(text_array):
            return [str(' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split()))
                    for text in text_array]

        text_pipe = Pipeline([('lower_case', FunctionTransformer(text_lowercase)),
                              ('remove_url', FunctionTransformer(remove_urls)),
                              ('remove_numbers', FunctionTransformer(remove_numbers)),
                              ('remove_punctuation', FunctionTransformer(remove_punctuation)),
                              ('vect', CountVectorizer(min_df=0.01, max_df=0.5)), ('tfidf', TfidfTransformer())])
        transformers.append(('text', text_pipe, text_columns[0]))
    for cat_column in categorical_columns:
        cat_pipe = Pipeline([('simpleimputer', SimpleImputer(strategy='most_frequent')),
                             ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))])
        transformers.append((f"cat_{cat_column}", cat_pipe, [cat_column]))

    featurization = ColumnTransformer(transformers)

    return featurization


def get_model():
    model = XGBClassifier(max_depth=12, tree_method='hist', n_jobs=1)

    return model


def main_function():
    # analysis = 'none'
    # analysis = 'data_corruption'
    analysis = 'data_cleaning'
    if len(sys.argv) > 1:
        analysis = sys.argv[1]
    if analysis == 'data_corruption':
        variant_count = 25
    elif analysis == 'data_cleaning':
        variant_count = 3  # FIXME: Add support for more
    elif analysis == 'none':
        variant_count = 1
    else:
        raise ValueError(f"Invalid analysis!")
    seed = 1234
    if len(sys.argv) > 3:
        seed = sys.argv[3]
    np.random.seed(seed)
    random.seed(seed)
    scores = []
    for variant_index in range(variant_count):
        print(f'Running fast_loading featurization featurization_3 on dataset reviews with model '
              f'xgboost with analysis {analysis} for variant {variant_index}')
        train, train_labels, test, test_labels, numerical_columns, categorical_columns, text_columns = \
            get_dataset(seed, analysis, variant_index)
        featurization = get_featurization(numerical_columns, categorical_columns, text_columns)
        model = get_model()
        pipeline = Pipeline([('featurization', featurization),
                             ('model', model)])
        pipeline = pipeline.fit(train, train_labels)
        predictions = pipeline.predict(test)
        score = accuracy_score(predictions, test_labels)
        print('    Score: ', score)
        scores.append(score)
    results = pd.DataFrame({'variant_index': range(variant_count), 'score': scores})
    print(results)


# Make sure this code is not executed during imports
if sys.argv[0] == "mlwhatif" or __name__ == "__main__":
    main_function()
