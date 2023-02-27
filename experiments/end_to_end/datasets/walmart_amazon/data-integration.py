# pylint: disable-all
import os

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import FunctionTransformer
from textdistance import levenshtein

from mlwhatif.utils import get_project_root

data_root = os.path.join(str(get_project_root()), "experiments", "end_to_end", "datasets")

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


train_data = join_and_prepare(train_pairs, table_a, table_b)
test_data = join_and_prepare(test_pairs, table_a, table_b)


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


lev = FunctionTransformer(levenshtein_distance)
le = FunctionTransformer(length_diff)

pipeline = Pipeline([
    ('columntransformer', ColumnTransformer([
        ('bert', FunctionTransformer(bert_distance), ['title_a', 'title_b']),
        ('levenshtein', FunctionTransformer(levenshtein_distance), ['title_a', 'title_b']),
        ('length', FunctionTransformer(length_diff), ['title_a', 'title_b']),
        ('categorical', OneHotEncoder(), ['category_match', 'brand_match']),
    ])),
    ('decisiontreeclassifier', DecisionTreeClassifier(random_state=0))])

model = pipeline.fit(train_data, train_data['label'])

score = model.score(test_data, test_data['label'])
print(f"score: {score}")
