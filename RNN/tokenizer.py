import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Embedding, GlobalAveragePooling1D
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

train_data_path = '/home/user/Documents/dataset/20_newsgroup/20news-bydate-train'
test_data_path = '/home/user/Documents/dataset/20_newsgroup/20news-bydate-test'
GLoVe_path = '/home1/dataset/GLoVe/glove.6B.100d.txt'

word_num = 20000
max_len = 1024
embedding_dim = 100

# training config
batch_size = 128
# train_num = 11314
train_num = 11270
iterations_per_epoch = int(train_num / batch_size)
epoch_num = 20

# test config
test_batch_size = 128
# test_num = 7532
test_num = 7503
test_iterations = int(test_num / test_batch_size)

def load_data(path):
    texts = []
    labels = []

    for i, label in enumerate(sorted(os.listdir(path))):
        for file_name in os.listdir(os.path.join(path, label)):
            file_path = os.path.join(path, label, file_name)
            try:
                with open(file_path, 'r') as f:
                    texts.append(f.read())
                    labels.append(i)
            except:
                print('can not decode:', file_path)
    return texts, labels

class Model(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)

        embedding_weight = self.get_embedding_weight(GLoVe_path, word_index)
        self.embedding = Embedding(word_num, embedding_dim, weights=[embedding_weight])
        self.conv1 = Conv1D(128, 5, activation='relu')
        self.pooling1 = MaxPooling1D(5)
        self.conv2 = Conv1D(128, 5, activation='relu')
        self.pooling2 = MaxPooling1D(5)
        self.conv3 = Conv1D(128, 5, activation='relu')
        self.global_pooling = GlobalAveragePooling1D()
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(20, activation='softmax')

    def call(self, inputs):
        net = self.embedding(inputs)
        net = self.conv1(net)
        net = self.pooling1(net)
        net = self.conv2(net)
        net = self.pooling2(net)
        net = self.conv3(net)
        net = self.global_pooling(net)
        net = self.fc1(net)
        net = self.fc2(net)
        return net

    def get_embedding_weight(self, weight_path, word_index):
        # embedding_weight = np.zeros([word_num, embedding_dim])
        embedding_weight = np.random.uniform(-0.05, 0.05, size=[word_num, embedding_dim])
        cnt = 0
        with open(weight_path, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word in word_index.keys() and word_index[word]< word_num:
                    weight = np.asarray(values[1:], dtype='float32')
                    embedding_weight[word_index[word]] = weight
                    cnt += 1
        print('word num: {}, matched num: {}'.format(len(word_index), cnt))
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
        y = tf.keras.utils.to_categorical(y, 20)

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
        y = tf.keras.utils.to_categorical(y, 20)

        loss, prediction = test_step(model, x, y)
        sum_loss += loss
        sum_accuracy += accuracy(y, prediction)

    print('test, loss:%f, accuracy:%f' %
          (sum_loss / test_iterations, sum_accuracy / test_iterations))


if __name__ == '__main__':
    # gpu config
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

    train_texts, train_labels = load_data(train_data_path)
    test_texts, test_labels = load_data(test_data_path)
    print('train num: {}, test num: {}'.format(len(train_texts), len(test_texts)))

    tokenizer = Tokenizer(num_words=word_num)
    tokenizer.fit_on_texts(train_texts)

    print('most common words:\nword rank')
    word_index = tokenizer.word_index
    for w, c in word_index.items():
        if c < 5:
            print(w, c)

    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)

    train_sequences = pad_sequences(train_sequences, maxlen=max_len)
    test_sequences = pad_sequences(test_sequences, maxlen=max_len)

    # get model
    model = Model()
    model.build(input_shape=(batch_size, max_len))

    # show
    model.summary()

    # train
    optimizer = optimizers.Adam()
    for epoch in range(epoch_num):
        print('epoch %d' % epoch)
        train(model, optimizer, train_sequences, train_labels)
        test(model, test_sequences, test_labels)