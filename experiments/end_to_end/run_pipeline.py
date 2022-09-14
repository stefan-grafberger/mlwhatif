import datetime
import os
import random
import sys

import numpy as np
import pandas as pd
from faker import Faker
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import label_binarize
from tensorflow.keras.layers import Dense  # pylint: disable=no-name-in-module
from tensorflow.keras.models import Sequential  # pylint: disable=no-name-in-module
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD  # pylint: disable=no-name-in-module
from xgboost import XGBClassifier

from example_pipelines.healthcare.healthcare_utils import MyKerasClassifier
from mlwhatif.utils import get_project_root

data_root = os.path.join(str(get_project_root()), "experiments", "end_to_end", "datasets")


def get_dataset(dataset_name, data_loading_name, seed):
    if dataset_name == 'reviews':
        def random_subset(arr):
            size = np.random.randint(low=1, high=len(arr) + 1)
            choice = np.random.choice(arr, size=size, replace=False)
            return [str(item) for item in choice]

        def load_data():
            if data_loading_name == 'fast_loading':
                reviews = pd.read_csv(os.path.join(data_root, "reviews", "reviews.csv.gz"), compression='gzip',
                                      index_col=0)
                ratings = pd.read_csv(os.path.join(data_root, "reviews", "ratings.csv"), index_col=0)
                products = pd.read_csv(os.path.join(data_root, "reviews", "products.csv"), index_col=0)
                categories = pd.read_csv(os.path.join(data_root, "reviews", "categories.csv"), index_col=0)
            elif data_loading_name == 'slow_loading':
                reviews = pd.read_csv("https://raw.githubusercontent.com/stefan-grafberger/ml-pipeline-datasets/"
                                      "main/datasets/reviews/reviews.csv.gz", compression='gzip', index_col=0)
                ratings = pd.read_csv("https://raw.githubusercontent.com/stefan-grafberger/ml-pipeline-datasets/"
                                      "main/datasets/reviews/ratings.csv", index_col=0)
                products = pd.read_csv("https://raw.githubusercontent.com/stefan-grafberger/ml-pipeline-datasets/"
                                       "main/datasets/reviews/products.csv", index_col=0)
                categories = pd.read_csv("https://raw.githubusercontent.com/stefan-grafberger/ml-pipeline-datasets/"
                                         "main/datasets/reviews/categories.csv", index_col=0)
            else:
                raise ValueError(f"Invalid data loading speed: {data_loading_name}!")

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
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}!")

    return train, train_labels, test, test_labels, numerical_columns, categorical_columns, text_columns


def get_featurization(featurization_name, numerical_columns, categorical_columns, text_columns):
    if featurization_name == 'fast':
        assert len(text_columns) == 1
        transformers = [('num', StandardScaler(), numerical_columns),
                        ('text', HashingVectorizer(n_features=2 ** 4), text_columns[0])]
        for cat_column in categorical_columns:
            transformers.append((f"cat_{cat_column}", OneHotEncoder(handle_unknown='ignore'), [cat_column]))

        featurization = ColumnTransformer(transformers)
    else:
        raise ValueError(f"Invalid featurization name: {featurization_name}!")

    return featurization


def get_model(model_name):
    if model_name == 'logistic_regression':
        model = SGDClassifier(loss='log')
    elif model_name == 'xgboost':
        model = XGBClassifier(max_depth=12, tree_method='hist')
    elif model_name == 'neural_network':
        def create_model(input_dim=10):
            """Create a simple neural network"""
            clf = Sequential()
            clf.add(Dense(16, kernel_initializer='normal', activation='relu', input_dim=input_dim))
            clf.add(Dense(8, kernel_initializer='normal', activation='relu'))
            clf.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
            clf.compile(loss='binary_crossentropy', optimizer=SGD(), metrics=["accuracy"])
            return clf

        model = MyKerasClassifier(build_fn=create_model, epochs=7, batch_size=32, verbose=0)
    else:
        raise ValueError(f"Invalid model name: {model_name}!")

    return model


# Make sure this code is not executed during imports
if sys.argv[0] == "mlwhatif" or __name__ == "__main__":

    dataset_name = 'reviews'
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]

    data_loading_name = 'fast'
    if len(sys.argv) > 2:
        data_loading_name = sys.argv[2]

    featurization_name = 'fast'
    if len(sys.argv) > 3:
        featurization_name = sys.argv[3]

    model_name = 'logistic_regression'
    if len(sys.argv) > 4:
        model_name = sys.argv[4]

    seed = 1234
    if len(sys.argv) > 5:
        seed = sys.argv[5]

    np.random.seed(seed)
    random.seed(seed)

    print(f'Running {data_loading_name} featurization {featurization_name} on dataset {dataset_name} with model '
          f'{model_name}')
    train, train_labels, test, test_labels, numerical_columns, categorical_columns, text_columns = \
        get_dataset(dataset_name, data_loading_name, seed)

    featurization = get_featurization(featurization_name, numerical_columns, categorical_columns, text_columns)
    model = get_model(model_name)

    pipeline = Pipeline([('featurization', featurization),
                         ('model', model)])

    pipeline = pipeline.fit(train, train_labels)
    predictions = pipeline.predict(test)
    print('    Score: ', accuracy_score(predictions, test_labels))
