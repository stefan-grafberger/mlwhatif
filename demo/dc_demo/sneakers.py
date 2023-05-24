import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import label_binarize
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

from mlwhatif.utils import get_project_root
from mlwhatif.utils._utils import decode_image

train_data = pd.read_csv(
    os.path.join(str(get_project_root()), "experiments", "end_to_end", "datasets", "sneakers",
                 "product_images.csv"), converters={'image': decode_image})
product_categories = pd.read_csv(
    os.path.join(str(get_project_root()), "experiments", "end_to_end", "datasets", "sneakers",
                 "product_categories.csv"))

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


def normalise_image(images):
    return images / 255.0

def reshape_images(images):
    return np.concatenate(images['image'].values) \
        .reshape(images.shape[0], 28, 28, 1)

featurization = Pipeline([
    ('normalisation', FunctionTransformer(normalise_image)),
    ('reshaping', FunctionTransformer(reshape_images))
])


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

pipeline = Pipeline([('featurization', featurization),
                     ('model', model)])
pipeline = pipeline.fit(train, train_labels)
predictions = pipeline.predict(test)
print('    Score: ', accuracy_score(test_labels, predictions))
