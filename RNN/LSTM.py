import tensorflow as tf
import numpy as np
import json
from tqdm import tqdm
import os
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

data_path = '/home1/dataset/IMDB/imdb.npz'
word_index_path = '/home1/dataset/IMDB/imdb_word_index.json'
GLoVe_path = '/home1/dataset/GLoVe/glove.6B.100d.txt'
word_num = 10000
max_len = 256
embedding_dim = 100

# training config
batch_size = 512
train_num = 25000
iterations_per_epoch = int(train_num / batch_size)
epoch_num = 10

# test config
test_batch_size = 500
test_num = 25000
test_iterations = int(test_num / test_batch_size)

class LSTMCell(tf.keras.layers.Layer):
    def __init__(self, output_dim, activation=tf.nn.tanh, forget_bias=1.0, **kwargs):
        self.output_dim = output_dim
        self.activation = activation
        self.forget_bias = forget_bias
        super(LSTMCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1] + self.output_dim, self.output_dim * 4),
                                      initializer=tf.initializers.glorot_uniform)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim * 4,),
                                    initializer=tf.initializers.zeros)

    def call(self, inputs, state):
        c, h = state
        net = tf.concat([inputs, h], axis=-1)
        net = tf.matmul(net, self.kernel) + self.bias

        i, j, f, o = tf.split(net, num_or_size_splits=4, axis=-1)
        new_c = (c * tf.sigmoid(f + self.forget_bias) + tf.sigmoid(i) * self.activation(j))
        new_h = self.activation(new_c) * tf.sigmoid(o)

        new_state = (new_c, new_h)
        return new_h, new_state

class LSTM(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.cell = LSTMCell(output_dim)
        super(LSTM, self).__init__(**kwargs)

    def call(self, inputs):
        inputs = tf.transpose(inputs, [1, 0, 2])
        # zero initial state
        state = (tf.constant(0.0, shape=[inputs.shape[1], self.output_dim]),
                 tf.constant(0.0, shape=[inputs.shape[1], self.output_dim]))

        output = []
        inputs = tf.unstack(inputs, axis=0)
        for i in range(len(inputs)):
            h, state = self.cell(inputs[i], state)
            output.append(h)
        output = tf.stack(output, axis=0)
        output = tf.transpose(output, [1, 0, 2])

        return output

class Model(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)

        with open(word_index_path, 'r') as f:
            word_index = json.load(f)
        embedding_weight = self.get_embedding_weight(GLoVe_path, word_index)
        self.embedding = Embedding(word_num, embedding_dim, weights=[embedding_weight])
        self.LSTM = LSTM(128)
        self.fc = Dense(2, activation='softmax')

    def call(self, inputs):
        net = self.embedding(inputs)
        net = self.LSTM(net)
        net = self.fc(net[:, -1, :])
        return net

    def get_embedding_weight(self, weight_path, word_index):
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

def cross_entropy(y_true, y_pred):
    cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(cross_entropy)

def accuracy(y_true, y_pred):
    correct_num = tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1))
    accuracy = tf.reduce_mean(tf.cast(correct_num, dtype=tf.float32))
    return accuracy

@tf.function
def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        prediction = model(x, training=True)
        loss = cross_entropy(y, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, prediction

@tf.function
def test_step(model, x, y):
    prediction = model(x, training=False)
    loss = cross_entropy(y, prediction)
    return loss, prediction

def train(model, optimizer, sequences, labels):
    sum_loss = 0
    sum_accuracy = 0

    # random shuffle
    seed = np.random.randint(0, 65536)
    np.random.seed(seed)
    np.random.shuffle(sequences)
    np.random.seed(seed)
    np.random.shuffle(labels)

    for i in tqdm(range(iterations_per_epoch)):
        x = sequences[i * batch_size: (i + 1) * batch_size, :]
        y = labels[i * batch_size: (i + 1) * batch_size]
        y = tf.keras.utils.to_categorical(y, 2)

        loss, prediction = train_step(model, optimizer, x, y)
        sum_loss += loss
        sum_accuracy += accuracy(y, prediction)

    print('ce_loss:%f, accuracy:%f' %
          (sum_loss / iterations_per_epoch, sum_accuracy / iterations_per_epoch))

def test(model, sequences, labels):
    sum_loss = 0
    sum_accuracy = 0

    for i in tqdm(range(test_iterations)):
        x = sequences[i * test_batch_size: (i + 1) * test_batch_size, :]
        y = labels[i * test_batch_size: (i + 1) * test_batch_size]
        y = tf.keras.utils.to_categorical(y, 2)

        loss, prediction = test_step(model, x, y)
        sum_loss += loss
        sum_accuracy += accuracy(y, prediction)

    print('test, loss:%f, accuracy:%f' %
          (sum_loss / test_iterations, sum_accuracy / test_iterations))


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
    model.build(input_shape=(batch_size, 256))

    # show
    model.summary()

    # train
    optimizer = optimizers.Adam()
    for epoch in range(epoch_num):
        print('epoch %d' % epoch)
        train(model, optimizer, train_sequences, train_labels)
        test(model, test_sequences, test_labels)

