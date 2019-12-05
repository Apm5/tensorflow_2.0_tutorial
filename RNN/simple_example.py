import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Embedding
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

data_path = '/home1/dataset/IMDB/imdb.npz'
word_index_path = '/home1/dataset/IMDB/imdb_word_index.json'
GLoVe_path = '/home1/dataset/GLoVe/glove.6B.100d.txt'
word_num = 10000
max_len = 256
embedding_dim = 100

def get_embedding_weight(weight_path, word_index):
    # embedding_weight = np.zeros([word_num, embedding_dim])
    embedding_weight = np.random.uniform(-0.05, 0.05, size=[word_num, embedding_dim])
    cnt = 0
    with open(weight_path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in word_index.keys() and word_index[word] + 3 < word_num:
                weight = np.asarray(values[1:], dtype='float32')
                embedding_weight[word_index[word] + 3] = weight
                cnt += 1
    print('matched word num: {}'.format(cnt))
    return embedding_weight

def Model():
    with open(word_index_path, 'r') as f:
        word_index = json.load(f)
    embedding_weight = get_embedding_weight(GLoVe_path, word_index)

    model = tf.keras.Sequential()
    model.add(Embedding(word_num, embedding_dim, weights=[embedding_weight]))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dense(2, activation='softmax'))
    return model

if __name__ == '__main__':
    # gpu config
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

    # load data
    imdb = tf.keras.datasets.imdb
    (train_sequences, train_labels), (test_sequences, test_labels) = imdb.load_data(data_path, num_words=word_num)

    train_sequences = pad_sequences(train_sequences, maxlen=max_len)
    test_sequences = pad_sequences(test_sequences, maxlen=max_len)

    # get model
    model = Model()
    model.summary()

    # train
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_sequences,
              train_labels,
              batch_size=512,
              epochs=10)

    # test
    test_loss, test_acc = model.evaluate(test_sequences, test_labels)
    print(test_acc)