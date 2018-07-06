import pickle
import pandas as pd
import numpy as np
import random
from collections import Counter

# loading the user posts
with open ('tokenized_formatted_data.txt', 'rb') as fp:
    posts = pickle.load(fp)

# reading the csv file in
df = pd.read_csv('mbti_1.csv')

# populating types array
types = []
for _type in df['type']:
    types.append(_type)

# getting the count of types in dataset
counts = Counter(types)

# shuffling the data
random.seed(673)
random.shuffle(posts)
random.seed(673)
random.shuffle(types)

# separating test and train data
train_types = types[1000:]
train_posts = posts[1000:]

def posts_generator(label):

    global train_types
    global train_posts

    # getting indices of instances of specific type
    indices = []
    for i in range(0, len(train_types)):
        if types[i] == label:
            indices.append(i)

    # getting average length of words used by users
    avg_length = 0
    for index in indices:
        avg_length += len(train_posts[index])

    avg_length = round(avg_length/len(indices))

    # making a pool of the words used by estj people
    all_words = []
    for index in indices:
        for word in train_posts[index]:
            if word not in all_words:
                all_words.append(word)

    # generate synthetic entj users
    new_users = []
    new_labels = []

    for i in range(counts[label], counts['INFP']):
        user = []
        for i in range(0, round(avg_length)):
            user.append(random.choice(all_words))

        new_users.append(user)
        new_labels.append(label)

    print (str(len(new_users)), "New Users of Type:", label)

    return (new_labels, new_users)

synthetic_data = []

for label in counts.keys():
    synthetic_data.append(posts_generator(label))

import pickle

with open('synthetic_data.pkl', 'wb') as fw:
    pickle.dump(synthetic_data, fw)
