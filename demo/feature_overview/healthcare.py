# pylint: disable-all
import os
import warnings

import numpy
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

from example_pipelines.healthcare.healthcare_utils import MyW2VTransformer, MyKerasClassifier

# Disable tensorflow API and optimization warnings for a readable output
warnings.filterwarnings('ignore')

seed = 1234
tensorflow.random.set_seed(seed)
numpy.seed = seed


def combine(patients, patient_histories, consent_required):
    if consent_required:
        patients = patients[patients['gave_consent'] == True]
    with_history = patients.merge(patient_histories, on="ssn")
    return with_history


def create_neural_net(input_dim):
    # Model definition
    nn = Sequential([
        Dense(8, activation='relu', input_dim=input_dim), Dropout(0.3),
        Dense(4, activation='relu'),
        Dense(2, activation='softmax')])
    nn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='accuracy')
    return nn


def featurization():
    # Featurization
    encode = ColumnTransformer(transformers=[
        ('numerical_features', StandardScaler(), ['weight']),
        ('categorical_features', OneHotEncoder(handle_unknown='ignore'), ['smokes']),
        ('textual_features', MyW2VTransformer(min_count=1, size=5), ['notes'])])
    return encode


def execute_pipeline():
    # Relational preprocessing
    patients = pd.read_csv("patients.csv")
    histories = pd.read_csv("histories.csv")
    test_histories = pd.read_csv("test_histories.csv")
    histories = histories[histories['hospital'].isin(["AL", "AK", "AR"])]
    train = combine(patients, histories, consent_required=True)
    test = combine(patients, test_histories, consent_required=True)

    # Model training and scoring
    encode_and_learn = Pipeline([
        ('features', featurization()),
        ('learner', MyKerasClassifier(create_neural_net, epochs=5, verbose=0))])
    model = encode_and_learn.fit(train, train['has_complication'])
    pred = model.predict(test)
    return accuracy_score(pred, test['has_complication'])


score = execute_pipeline()
print(f"Score: {score}")
