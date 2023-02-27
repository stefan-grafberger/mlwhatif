# pylint: disable-all
import os

# import requests
import re
import pandas as pd
import json
import time

from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer

from mlwhatif.utils import get_project_root

data_root = os.path.join(str(get_project_root()), "experiments", "end_to_end", "datasets")


def get_reddit(subreddit, after=None):
    source_file = os.path.join(data_root, "reddit", "data",  f'reddit-{subreddit}.txt')
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

# There are better, more expensive models: https://www.sbert.net/docs/pretrained_models.html
model = SentenceTransformer('all-MiniLM-L6-v2')

encoder = FunctionTransformer(lambda row: model.encode(row['text'].values))

X_train = encoder.transform(train_data)
X_test = encoder.transform(test_data)

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

def create_model(input_dim=384):
    clf = Sequential()
    clf.add(Dense(16, activation='relu', input_dim=input_dim))
    clf.add(Dense(8, activation='relu'))
    clf.add(Dense(2, activation='softmax'))
    clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=["accuracy"])
    return clf

neuralnet = KerasClassifier(build_fn=create_model, epochs=20, batch_size=64, verbose=1)

neuralnet.fit(X_train, train_data['label'])

score = neuralnet.score(X_test, test_data['label'])
print(f"score: {score}")
