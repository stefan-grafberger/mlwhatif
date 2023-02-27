# pylint: disable-all
import datetime
import json
import os
import random
import re
import string
import sys
import time

import fuzzy_pandas
import numpy as np
import pandas as pd
from faker import Faker
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, RobustScaler
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD  # pylint: disable=no-name-in-module
from textdistance import levenshtein
from xgboost import XGBClassifier

from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.models import Sequential

from example_pipelines.healthcare.healthcare_utils import MyKerasClassifier, MyW2VTransformer
from mlmq.utils import get_project_root
from mlmq.utils._utils import decode_image

data_root = os.path.join(str(get_project_root()), "experiments", "end_to_end", "datasets")


def get_dataset(dataset_name, data_loading_name, seed, featurization_name):
    if dataset_name in {'reviews', 'reviews_1x', 'reviews_5x', 'reviews_10x'}:
        def random_subset(arr):
            size = np.random.randint(low=1, high=len(arr) + 1)
            choice = np.random.choice(arr, size=size, replace=False)
            return [str(item) for item in choice]

        if dataset_name in {'reviews', 'reviews_1x'}:
            def load_data():
                if data_loading_name == 'fast_loading':
                    reviews = pd.read_csv(os.path.join(data_root, "reviews", "reviews.csv.gz"), compression='gzip',
                                          index_col=0)
                    ratings = pd.read_csv(os.path.join(data_root, "reviews", "ratings.csv"), index_col=0)
                    products = pd.read_csv(os.path.join(data_root, "reviews", "products.csv"), index_col=0)
                    categories = pd.read_csv(os.path.join(data_root, "reviews", "categories.csv"), index_col=0)
                elif data_loading_name == 'slow_loading':
                    reviews = pd.read_csv("https://raw.githubusercontent.com/anonymous-52200/ml-pipeline-datasets/"
                                          "main/datasets/reviews/reviews.csv.gz", compression='gzip', index_col=0)
                    ratings = pd.read_csv("https://raw.githubusercontent.com/anonymous-52200/ml-pipeline-datasets/"
                                          "main/datasets/reviews/ratings.csv", index_col=0)
                    products = pd.read_csv("https://raw.githubusercontent.com/anonymous-52200/ml-pipeline-datasets/"
                                           "main/datasets/reviews/products.csv", index_col=0)
                    categories = pd.read_csv("https://raw.githubusercontent.com/anonymous-52200/ml-pipeline-datasets/"
                                             "main/datasets/reviews/categories.csv", index_col=0)
                else:
                    raise ValueError(f"Invalid data loading speed: {data_loading_name}!")

                return reviews, ratings, products, categories
        elif dataset_name == 'reviews_5x':
            def load_data():
                if data_loading_name == 'fast_loading':
                    reviews = pd.read_csv(os.path.join(data_root, "reviews_large", "reviews_500000.csv.gz"),
                                          compression='gzip', index_col=0)
                    ratings = pd.read_csv(os.path.join(data_root, "reviews_large", "ratings_500000.csv"), index_col=0)
                    products = pd.read_csv(os.path.join(data_root, "reviews_large", "products_500000.csv"), index_col=0)
                    categories = pd.read_csv(os.path.join(data_root, "reviews_large", "categories_500000.csv"),
                                             index_col=0)
                else:
                    raise ValueError(f"Invalid data loading speed: {data_loading_name}!")

                return reviews, ratings, products, categories
        elif dataset_name == 'reviews_10x':
            def load_data():
                if data_loading_name == 'fast_loading':
                    reviews = pd.read_csv(os.path.join(data_root, "reviews_large", "reviews_1000000.csv.gz"),
                                          compression='gzip', index_col=0)
                    ratings = pd.read_csv(os.path.join(data_root, "reviews_large", "ratings_1000000.csv"), index_col=0)
                    products = pd.read_csv(os.path.join(data_root, "reviews_large", "products_1000000.csv"),
                                           index_col=0)
                    categories = pd.read_csv(os.path.join(data_root, "reviews_large", "categories_1000000.csv"),
                                             index_col=0)
                else:
                    raise ValueError(f"Invalid data loading speed: {data_loading_name}!")

                return reviews, ratings, products, categories

        else:
            raise ValueError(f"Invalid dataset_name: {dataset_name}!")

        if dataset_name == 'reviews':
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
        else:
            def integrate_data(reviews, ratings, products, categories, fake):
                start_date = fake.date_between(start_date=datetime.date(year=1900, month=1, day=1),
                                               end_date=datetime.date(year=1905, month=6, day=1))

                reviews = reviews[reviews['review_date'] >= start_date.strftime('%Y-%m-%d')]

                reviews_with_ratings = reviews.merge(ratings, on='review_id')
                products_with_categories = products.merge(left_on='category_id', right_on='id', right=categories)

                # random_categories = random_subset(list(categories.category))
                # print(f"random_categories: {random_categories}", file=sys.stderr)
                # products_with_categories = products_with_categories[
                #     products_with_categories['category'].isin(random_categories)]
                reviews_with_products_and_ratings = reviews_with_ratings.merge(products_with_categories,
                                                                               on='product_id')

                return reviews_with_products_and_ratings

            def compute_feature_and_label_data(reviews_with_products_and_ratings, final_columns, fake):
                reviews_with_products_and_ratings['is_helpful'] = reviews_with_products_and_ratings['helpful_votes'] > 0

                projected_reviews = reviews_with_products_and_ratings[final_columns]

                train_data, test_data = train_test_split(projected_reviews)

                train_labels = label_binarize(train_data['is_helpful'], classes=[True, False]).ravel()
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
        print(f"len(train), len(test): {len(train), len(test)}", file=sys.stderr)
    elif dataset_name == 'healthcare':
        COUNTIES_OF_INTEREST = ['county2', 'county3']

        if data_loading_name == 'fast_loading':
            patients = pd.read_csv(
                os.path.join(str(get_project_root()), "experiments", "end_to_end", "datasets", "healthcare",
                             "patients.csv"), na_values='?')
            histories = pd.read_csv(
                os.path.join(str(get_project_root()), "experiments", "end_to_end", "datasets", "healthcare",
                             "histories.csv"), na_values='?')
        elif data_loading_name == 'slow_loading':
            patients = pd.read_csv("https://raw.githubusercontent.com/anonymous-52200/ml-pipeline-datasets/"
                                   "main/datasets/healthcare/patients.csv", na_values='?')
            histories = pd.read_csv("https://raw.githubusercontent.com/anonymous-52200/ml-pipeline-datasets/"
                                    "main/datasets/healthcare/histories.csv", na_values='?')
        else:
            raise ValueError(f"Invalid data loading speed: {data_loading_name}!")

        data = fuzzy_pandas.fuzzy_merge(patients, histories, on='full_name', method='levenshtein',
                                        keep_right=['smoker', 'complications'], threshold=0.95)
        complications = data.groupby('age_group') \
            .agg(mean_complications=('complications', 'mean'))
        data = data.merge(complications, on=['age_group'])
        data['label'] = data['complications'] > 1.2 * data['mean_complications']
        data = data[['smoker', 'last_name', 'county', 'num_children', 'race', 'income', 'label']]
        data = data[data['county'].isin(COUNTIES_OF_INTEREST)]
        numerical_columns = ['num_children', 'income']
        categorical_columns = ['smoker', 'county', 'race']
        text_columns = ['last_name']
        train, test = train_test_split(data)
        if featurization_name == "featurization_0":
            train = train.dropna(subset=categorical_columns)
            test = test.dropna(subset=categorical_columns)
        train_labels = train['label']
        test_labels = test['label']
    elif dataset_name in {'folktables', 'folktables_1x', 'folktables_5x', 'folktables_10x'}:
        if dataset_name == 'folktables':
            if data_loading_name == 'fast_loading':
                acs_data = pd.read_csv(
                    os.path.join(str(get_project_root()), "experiments", "end_to_end", "datasets", "folktables",
                                 "acs_income_RI_2017_5y.csv"), delimiter=";")
            elif data_loading_name == 'slow_loading':
                acs_data = pd.read_csv("https://raw.githubusercontent.com/anonymous-52200/ml-pipeline-datasets/"
                                       "main/datasets/folktables/acs_income_RI_2017_5y.csv", delimiter=";")
            else:
                raise ValueError(f"Invalid data loading speed: {data_loading_name}!")
        elif dataset_name == 'folktables_1x':
            if data_loading_name == 'fast_loading':
                acs_data = pd.read_csv(
                    os.path.join(str(get_project_root()), "experiments", "end_to_end", "datasets", "folktables_large",
                                 "acs_income_all_2017_1y_100000.csv"), delimiter=";")
            else:
                raise ValueError(f"Invalid data loading speed: {data_loading_name}!")
        elif dataset_name == 'folktables_5x':
            if data_loading_name == 'fast_loading':
                acs_data = pd.read_csv(
                    os.path.join(str(get_project_root()), "experiments", "end_to_end", "datasets", "folktables_large",
                                 "acs_income_all_2017_1y_500000.csv"), delimiter=";")
            else:
                raise ValueError(f"Invalid data loading speed: {data_loading_name}!")
        elif dataset_name == 'folktables_10x':
            if data_loading_name == 'fast_loading':
                acs_data = pd.read_csv(
                    os.path.join(str(get_project_root()), "experiments", "end_to_end", "datasets", "folktables_large",
                                 "acs_income_all_2017_1y_1000000.csv"), delimiter=";")
            else:
                raise ValueError(f"Invalid data loading speed: {data_loading_name}!")
        else:
            raise ValueError(f"Invalid dataset_name: {dataset_name}!")
        columns = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P', 'PINCP']
        acs_data = acs_data[columns]

        numerical_columns = ['AGEP', 'WKHP']
        categorical_columns = ['COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP']
        text_columns = []
        train, test = train_test_split(acs_data)
        if featurization_name == "featurization_0":
            train = train.dropna(subset=categorical_columns)
            test = test.dropna(subset=categorical_columns)
        train_labels = train['PINCP']
        test_labels = test['PINCP']
        print(f"len(train), len(test): {len(train)}, {len(test)}", file=sys.stderr)
    elif dataset_name == 'cardio':
        if data_loading_name == 'fast_loading':
            cardio_main = pd.read_csv(
                os.path.join(str(get_project_root()), "experiments", "end_to_end", "datasets", "cardio",
                             "cardio-main.csv"), delimiter=';')
            cardio_first_additional_table = pd.read_csv(
                os.path.join(str(get_project_root()), "experiments", "end_to_end", "datasets", "cardio",
                             "cardio-add-one.csv"), delimiter=';')
            cardio_second_additional_table = pd.read_csv(
                os.path.join(str(get_project_root()), "experiments", "end_to_end", "datasets", "cardio",
                             "cardio-add-two.csv"), delimiter=';')
        elif data_loading_name == 'slow_loading':
            cardio_main = pd.read_csv("https://raw.githubusercontent.com/anonymous-52200/ml-pipeline-datasets/"
                                      "main/datasets/cardio/cardio-main.csv", delimiter=';')
            cardio_first_additional_table = pd.read_csv("https://raw.githubusercontent.com/anonymous-52200/"
                                                        "ml-pipeline-datasets/main/datasets/cardio/cardio-add-one.csv",
                                                        delimiter=';')
            cardio_second_additional_table = pd.read_csv("https://raw.githubusercontent.com/anonymous-52200/"
                                                         "ml-pipeline-datasets/main/datasets/cardio/cardio-add-two.csv",
                                                         delimiter=';')
        else:
            raise ValueError(f"Invalid data loading speed: {data_loading_name}!")
        main_and_one = cardio_main.merge(cardio_first_additional_table, on="id")
        cardio_data = main_and_one.merge(cardio_second_additional_table, on="id")
        numerical_columns = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
        categorical_columns = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        text_columns = []

        columns = numerical_columns + categorical_columns + ["cardio"]
        cardio_data = cardio_data[columns]

        train, test = train_test_split(cardio_data, test_size=0.20, random_state=42)
        if featurization_name == "featurization_0":
            train = train.dropna(subset=categorical_columns)
            test = test.dropna(subset=categorical_columns)
        train_labels = train['cardio']
        test_labels = test['cardio']
    elif dataset_name == 'sneakers':
        if data_loading_name == 'fast_loading':
            train_data = pd.read_csv(
                os.path.join(str(get_project_root()), "experiments", "end_to_end", "datasets", "sneakers",
                             "product_images.csv"), converters={'image': decode_image})
            product_categories = pd.read_csv(
                os.path.join(str(get_project_root()), "experiments", "end_to_end", "datasets", "sneakers",
                             "product_categories.csv"))
        elif data_loading_name == 'slow_loading':
            train_data = pd.read_csv("https://raw.githubusercontent.com/anonymous-52200/ml-pipeline-datasets/"
                                     "main/datasets/sneakers/product_images.csv", converters={'image': decode_image})
            product_categories = pd.read_csv("https://raw.githubusercontent.com/anonymous-52200/"
                                             "ml-pipeline-datasets/main/datasets/sneakers/"
                                             "product_categories.csv")
        else:
            raise ValueError(f"Invalid data loading speed: {data_loading_name}!")
        with_categories = train_data.merge(product_categories, on='category_id')

        categories_to_distinguish = ['Sneaker', 'Ankle boot']

        images_of_interest = with_categories[with_categories['category_name'].isin(categories_to_distinguish)]

        random_seed_for_splitting = 1337

        train_with_labels, test_with_labels = train_test_split(
            images_of_interest, test_size=0.2, random_state=random_seed_for_splitting)

        train = train_with_labels[['image']]
        test = test_with_labels[['image']]
        train_labels = label_binarize(train_with_labels['category_name'], classes=categories_to_distinguish)
        test_labels = label_binarize(test_with_labels['category_name'], classes=categories_to_distinguish)
        numerical_columns = ['image']
        categorical_columns = []
        text_columns = []
    elif dataset_name == 'reddit':
        if data_loading_name == 'fast_loading':
            def get_reddit(subreddit, after=None):
                source_file = os.path.join(data_root, "reddit", "data", f'reddit-{subreddit}.txt')
                base_url = f'https://www.reddit.com/r/{subreddit}/controversial.json?limit=100&t=year'

                if after is not None:
                    base_url += f'&after=t3_{after}'
                    source_file = os.path.join(data_root, "reddit", "data", f'reddit-{subreddit}_{after}.txt')

                print(f'Loading posts from {base_url}')
                # request = requests.get(base_url, headers = {'User-agent': 'development'})
                # response = request.json()
                time.sleep(0.1)
                with open(source_file, "r") as infile:
                    return json.load(infile)

            raw_posts = {}
            for subreddit in ['bullying', 'selfimprovement', 'depression', 'FengShui']:
                raw_posts[subreddit] = []
                last_identifier = None
                for _ in range(0, 10):
                    raw_response = get_reddit(subreddit, after=last_identifier)

                    for post in raw_response['data']['children']:
                        raw_posts[subreddit].append(post)
                        last_identifier = post['data']['id']

            subreddits = ['bullying', 'selfimprovement', 'depression', 'FengShui']

            prepared_sentences = []
            for subreddit in subreddits:
                for num, post in enumerate(raw_posts[subreddit]):
                    identifier = post['data']['id']
                    text = post['data']['selftext']
                    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)
                    for index, sentence in enumerate(sentences):
                        sentence_id = identifier + '_' + str(index)
                        prepared_sentences.append((sentence_id, sentence, subreddit))
            labeled_sentences = pd.DataFrame.from_records(prepared_sentences, columns=['id', 'text', 'origin'])

            url_regex = '((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*'
            filtered_sentences = labeled_sentences[labeled_sentences['text'].str.len() > 20]
            filtered_sentences = filtered_sentences[filtered_sentences['text'].str.match(url_regex) != True]

            filtered_sentences['label'] = filtered_sentences['origin'].isin(['bullying', 'depression'])

            train_data, test_data = train_test_split(filtered_sentences, test_size=0.2, random_state=42, shuffle=True)
            train = train_data[['text']]
            test = test_data[['text']]
            train_labels = train_data['label']
            test_labels = test_data['label']
            numerical_columns = []
            categorical_columns = []
            text_columns = []
        else:
            raise ValueError(f"Invalid data loading speed: {data_loading_name}!")
    elif dataset_name == 'walmart_amazon':
        if data_loading_name == 'fast_loading':
            table_a = pd.read_csv(os.path.join(data_root, "walmart_amazon", "data", 'tableA.csv'))
            table_b = pd.read_csv(os.path.join(data_root, "walmart_amazon", "data", 'tableB.csv'))

            train_pairs = pd.read_csv(os.path.join(data_root, "walmart_amazon", "data", 'train.csv')).sample(frac=0.01)
            test_pairs = pd.read_csv(os.path.join(data_root, "walmart_amazon", "data", 'test.csv'))

            def join_and_prepare(pairs, table_a, table_b):
                data = pairs.merge(table_a, left_on='ltable_id', right_on='id')
                data = data.merge(table_b, left_on='rtable_id', right_on='id', suffixes=['_a', '_b'])
                data = data.fillna('')
                data['category_match'] = data['category_a'] == data['category_b']
                data['brand_match'] = data['brand_a'] == data['brand_b']
                return data

            train = join_and_prepare(train_pairs, table_a, table_b)
            test = join_and_prepare(test_pairs, table_a, table_b)

            train_labels = train['label']
            test_labels = test['label']
            numerical_columns = []
            categorical_columns = []
            text_columns = []
        else:
            raise ValueError(f"Invalid data loading speed: {data_loading_name}!")
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}!")

    return train, train_labels, test, test_labels, numerical_columns, categorical_columns, text_columns


def get_featurization(featurization_name, numerical_columns, categorical_columns, text_columns):
    if featurization_name == 'featurization_0':  # num preprocessing based on dspipes num_pipe_0
        def identity(x):
            return x

        transformers = [('num', FunctionTransformer(identity), numerical_columns)]

        if len(text_columns) >= 1:
            assert len(text_columns) == 1
            transformers.append(('text', HashingVectorizer(n_features=2 ** 4), text_columns[0]))

        # TODO: This works too, if there was a bug at some point with this its fixed now. Change this back?
        # transformers = [('num', FunctionTransformer(identity), numerical_columns),
        #                 ('text', HashingVectorizer(n_features=2 ** 4), text_columns[0]),
        #                 ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_columns)]

        for cat_column in categorical_columns:
            # Pipelines with categorical missing values need to, e.g., call dropna during dataset loading for
            # featurization_0 which has no missing value imputation. The other featurization options have imputation.
            transformers.append((f"cat_{cat_column}", OneHotEncoder(handle_unknown='ignore', sparse=False),
                                 [cat_column]))

        featurization = ColumnTransformer(transformers)
    elif featurization_name == 'featurization_1':  # based on openml_id == '5055' and openml_id == '17322'
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
    elif featurization_name == 'featurization_2':  # based on openml_id == '8774' and dspipes mode == 'txt_pipe_0'
        num_pipe = Pipeline([('imputer', SimpleImputer(add_indicator=True)),
                             ('standardscaler', StandardScaler())])
        transformers = [('num', num_pipe, numerical_columns)]
        if len(text_columns) >= 1:
            assert len(text_columns) == 1
            text_pipe = Pipeline([('vect', CountVectorizer(min_df=0.01, max_df=0.5)),
                                  ('tfidf', TfidfTransformer())])
            transformers.append(('text', text_pipe, text_columns[0]))
        for cat_column in categorical_columns:
            cat_pipe = Pipeline([('simpleimputer', SimpleImputer(strategy='most_frequent')),
                                 ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))])
            transformers.append((f"cat_{cat_column}", cat_pipe, [cat_column]))

        featurization = ColumnTransformer(transformers)
    elif featurization_name == 'featurization_3':  # based on dspipes 'num_pipe_3' and 'txt_pipe_2'
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
    elif featurization_name == 'featurization_4':  # based on healthcare, openml_id == '17322', dspipes 'num_pipe_3'
        union = FeatureUnion([("pca", PCA(n_components=(len(numerical_columns) - 1))),
                              ("svd", TruncatedSVD(n_iter=1, n_components=(len(numerical_columns) - 1)))])
        num_pipe = Pipeline([('union', union), ('scaler', StandardScaler())])

        transformers = [('num', num_pipe, numerical_columns)]
        if len(text_columns) >= 1:
            assert len(text_columns) == 1
            transformers.append(('text', MyW2VTransformer(min_count=1), text_columns))

        def another_imputer(df_with_categorical_columns):
            return df_with_categorical_columns.fillna('__missing__').astype(str)

        for cat_column in categorical_columns:
            cat_pipe = Pipeline([('anothersimpleimputer', FunctionTransformer(another_imputer)),
                                 ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))])
            transformers.append((f"cat_{cat_column}", cat_pipe, [cat_column]))

        featurization = ColumnTransformer(transformers)
    elif featurization_name == 'image':
        def normalise_image(images):
            return images / 255.0

        def reshape_images(images):
            return np.concatenate(images['image'].values) \
                .reshape(images.shape[0], 28, 28, 1)
        featurization = Pipeline([
            ('normalisation', FunctionTransformer(normalise_image)),
            ('reshaping', FunctionTransformer(reshape_images))
        ])
    elif featurization_name == 'reddit':
        model = SentenceTransformer('all-MiniLM-L6-v2')
        featurization = FunctionTransformer(lambda row: model.encode(row['text'].values))
    elif featurization_name == 'walmart_amazon':
        def bert_distance(df):
            # There are better, more expensive models: https://www.sbert.net/docs/pretrained_models.html
            bert = SentenceTransformer('all-MiniLM-L6-v2')
            column_a, column_b = df.columns
            encodings_a = bert.encode(df[column_a])
            encodings_b = bert.encode(df[column_b])
            distances = np.linalg.norm(encodings_a - encodings_b, axis=1)
            distances = distances.reshape((len(df), 1))
            return distances

        def levenshtein_distance(df):
            column_a, column_b = df.columns
            df['levenshtein'] = df.apply(lambda row: levenshtein.distance(row[column_a], row[column_b]), axis=1)
            distances = df['levenshtein'].values
            distances = distances.reshape((len(df), 1))
            return distances

        def length_diff(df):
            column_a, column_b = df.columns
            df['len'] = df.apply(lambda row: abs(len(row[column_a]) - len(row[column_b])), axis=1)
            distances = df['len'].values
            distances = distances.reshape((len(df), 1))
            return distances

        featurization = ColumnTransformer([
            ('bert', FunctionTransformer(bert_distance), ['title_a', 'title_b']),
            ('levenshtein', FunctionTransformer(levenshtein_distance), ['title_a', 'title_b']),
            ('length', FunctionTransformer(length_diff), ['title_a', 'title_b']),
            ('categorical', OneHotEncoder(), ['category_match', 'brand_match']),
        ])
    else:
        raise ValueError(f"Invalid featurization name: {featurization_name}!")

    return featurization


def get_model(model_name):
    if model_name == 'logistic_regression':
        model = SGDClassifier(loss='log', max_iter=30, n_jobs=1)
    elif model_name == 'xgboost':
        model = XGBClassifier(max_depth=12, tree_method='hist', n_jobs=1)
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
    elif model_name == 'image':
        def create_cnn():
            model = Sequential([
                Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)),
                MaxPooling2D(pool_size=2),
                Dropout(0.3),
                Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
                MaxPooling2D(pool_size=2),
                Dropout(0.3),
                Flatten(),
                Dense(256, activation='relu'),
                Dropout(0.5),
                Dense(2, activation='softmax')
            ])
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model
        model = KerasClassifier(create_cnn, epochs=10, verbose=0)
    elif model_name == 'reddit':
        def create_model(input_dim=384):
            clf = Sequential()
            clf.add(Dense(16, activation='relu', input_dim=input_dim))
            clf.add(Dense(8, activation='relu'))
            clf.add(Dense(2, activation='softmax'))
            clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=["accuracy"])
            return clf

        model = KerasClassifier(build_fn=create_model, epochs=20, batch_size=64, verbose=1)
    elif model_name == 'walmart_amazon':
        model = XGBClassifier(max_depth=12, tree_method='hist', n_jobs=1)
    else:
        raise ValueError(f"Invalid model name: {model_name}!")

    return model


def main_function(*args, **kwargs):
    dataset_name = 'reviews'
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    data_loading_name = 'fast_loading'
    if len(sys.argv) > 2:
        data_loading_name = sys.argv[2]
    featurization_name = 'featurization_0'
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
    if dataset_name == "sneakers":
        assert featurization_name == "image"
        assert model_name == "image"
    print(f'Running {data_loading_name} featurization {featurization_name} on dataset {dataset_name} with model '
          f'{model_name}')
    train, train_labels, test, test_labels, numerical_columns, categorical_columns, text_columns = \
        get_dataset(dataset_name, data_loading_name, seed, featurization_name)
    featurization = get_featurization(featurization_name, numerical_columns, categorical_columns, text_columns)
    model = get_model(model_name)
    pipeline = Pipeline([('featurization', featurization),
                         ('model', model)])
    pipeline = pipeline.fit(train, train_labels)
    predictions = pipeline.predict(test)
    print('    Score: ', accuracy_score(test_labels, predictions))


# Make sure this code is not executed during imports
if sys.argv[0] == "mlmq" or __name__ == "__main__":
    main_function()
