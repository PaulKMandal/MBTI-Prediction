def data():
    import pickle

    with open('embedding_labels.pkl', 'rb') as lf:
        labels = pickle.load(lf)

    with open('embedding_posts.pkl', 'rb') as pf:
        posts = pickle.load(pf)

    len(labels)

    # Tokenizing data
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing import sequence

    max_features = 10000
    max_len = 50
    batch_size = 32
    train_samples = 329885 # Roughly about 70% of Data
    max_words = 10000

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(posts)
    sequences = tokenizer.texts_to_sequences(posts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))

    data = sequence.pad_sequences(sequences, maxlen=max_len)
    label_dictionary = {
        'I': 0,
        'E': 1
    }

    labels = [label_dictionary[label[:1]] for label in labels]

    import numpy as np
    labels = np.asarray(labels)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    x_test = data[train_samples:]
    y_test = labels[train_samples:]

    x_train = data[:train_samples]
    y_train = labels[:train_samples]

    return x_train, y_train, x_test, y_test

class_weights = {
    0:1.0,
    1:3.27
}

from keras import models
from keras import layers
from keras.regularizers import l1_l2
from keras.layers import Embedding, Dropout, LSTM, Dense

from hyperas import optim
from hyperas.distributions import choice, uniform

def create_model(x_train, y_train, x_test, y_test):
    model = models.Sequential()
    model.add(Embedding(max_features, max_len))
    model.add(LSTM({{choice([8,16,32])}}, return_sequences=True))
    model.add(LSTM({{choice([8,16,32])}}))
    if {{choice(['three','two'])}} == 'three':
        model.add(Dense({{choice([8,16,32])}}))
    else:
        pass
              
    model.add(Dense(1, activation='sigmoid'))

    # model.layers[0].set_weights([embedding_matrix])
    # model.layers[0].trainable = False

    model.compile(optimizer={{choice(['rmsprop','adam','sgd'])}},
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
              
    model.fit(x_train, y_train, batch_size={{choice([64, 128])}},
             epochs={{uniform(10, 20)}}, validation_data=(x_test, y_test),
             class_weight=class_weights)
              
    score, acc = model.evaluate(x_test_, y_test)
    print('Test accuracy:', acc)
    return {'loss':-acc, 'status': STATUS_OK, 'model': model}

from hyperopt import Trials, STATUS_OK, tpe

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=30,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evaluation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)