import os
import random

import numpy as np
import pandas as pd
from faker import Faker
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
                 "product_images.csv"))
img_ids = list(range(0, len(train_data["category_id"])))
img_ids = [f"img_{id}" for id in img_ids]
train_data["image_id"] = img_ids
train_data.to_csv(os.path.join(str(get_project_root()), "demo", "dc_demo",
                 "product_images.csv"), index=False)
test_loading = pd.read_csv(
    os.path.join(str(get_project_root()), "demo", "dc_demo",
                 "product_images.csv"), converters={'image': decode_image})
print(test_loading)

# Create an instance of the Faker class
faker = Faker()

# Shuffle the list of IDs
random.shuffle(img_ids)

# Open the log file for writing
with open(os.path.join(str(get_project_root()), "demo", "dc_demo",
                 "log_file.txt"), 'w') as file:
    for id in img_ids[:5713]:
        # Generate a fake log message with the chosen ID
        log_message = f'{id}: {faker.sentence()}'

        # Generate a random timestamp
        timestamp = faker.date_time_between(start_date='-1w', end_date='now')

        # Format the timestamp as a string
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')

        # Construct the log entry with the timestamp
        log_entry = f'[{timestamp_str}] {log_message}'

        # Write the log entry to the file
        file.write(log_entry + '\n')
