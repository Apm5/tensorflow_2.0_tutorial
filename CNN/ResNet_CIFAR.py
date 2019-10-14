import tensorflow as tf
import numpy as np
import pickle as p
from tqdm import tqdm
import os
import cv2
import time
from tensorflow.keras import models, optimizers, regularizers
from tensorflow.keras.layers import Conv2D, AveragePooling2D, BatchNormalization, Flatten, Dense, Input, add, Activation
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# network config
stack_n = 18  # layers = stack_n * 6 + 2
weight_decay = 1e-4

# training config
batch_size = 128
train_num = 50000
iterations_per_epoch = int(train_num / batch_size)
learning_rate = [0.1, 0.01, 0.001]
boundaries = [80 * iterations_per_epoch, 120 * iterations_per_epoch]
epoch_num = 200

# test config
test_batch_size = 200
test_num = 10000
test_iterations = int(test_num / test_batch_size)

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb')as f:
        datadict = p.load(f, encoding='iso-8859-1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y


def load_CIFAR(Foldername):
    train_data = np.zeros([50000, 32, 32, 3], dtype=np.float32)
    train_label = np.zeros([50000, 10], dtype=np.float32)
    test_data = np.zeros([10000, 32, 32, 3], dtype=np.float32)
    test_label = np.zeros([10000, 10], dtype=np.float32)

    for sample in range(5):
        X, Y = load_CIFAR_batch(Foldername + "/data_batch_" + str(sample + 1))

        for i in range(3):
            train_data[10000 * sample:10000 * (sample + 1), :, :, i] = X[:, i, :, :]
        for i in range(10000):
            train_label[i + 10000 * sample][Y[i]] = 1

    X, Y = load_CIFAR_batch(Foldername + "/test_batch")
    for i in range(3):
        test_data[:, :, :, i] = X[:, i, :, :]
    for i in range(10000):
        test_label[i][Y[i]] = 1

    return train_data, train_label, test_data, test_label

def color_normalize(train_images, test_images):
    mean = [np.mean(train_images[:, :, :, i]) for i in range(3)]  # [125.307, 122.95, 113.865]
    std = [np.std(train_images[:, :, :, i]) for i in range(3)]  # [62.9932, 62.0887, 66.7048]
    for i in range(3):
        train_images[:, :, :, i] = (train_images[:, :, :, i] - mean[i]) / std[i]
        test_images[:, :, :, i] = (test_images[:, :, :, i] - mean[i]) / std[i]
    return train_images, test_images

def images_augment(images):
    output = []
    for img in images:
        img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        x = np.random.randint(0, 8)
        y = np.random.randint(0, 8)
        if np.random.randint(0, 2):
            img = cv2.flip(img, 1)
        output.append(img[x: x+32, y:y+32, :])
    return np.ascontiguousarray(output, dtype=np.float32)


def residual_block(inputs, channels, strides=(1, 1)):
    if strides == (1, 1):
        shortcut = inputs
    else:
        shortcut = Conv2D(channels, (1, 1), strides=strides)(inputs)

    net = BatchNormalization(momentum=0.9, epsilon=1e-5)(inputs)
    net = Activation('relu')(net)
    net = Conv2D(channels, (3, 3), padding='same', strides=strides)(net)
    net = BatchNormalization(momentum=0.9, epsilon=1e-5)(net)
    net = Activation('relu')(net)
    net = Conv2D(channels, (3, 3), padding='same')(net)

    net = add([net, shortcut])
    return net

def ResNet(inputs):
    net = Conv2D(16, (3, 3), padding='same')(inputs)

    for i in range(stack_n):
        net = residual_block(net, 16)

    net = residual_block(net, 32, strides=(2, 2))
    for i in range(stack_n - 1):
        net = residual_block(net, 32)

    net = residual_block(net, 64, strides=(2, 2))
    for i in range(stack_n - 1):
        net = residual_block(net, 64)

    net = BatchNormalization(momentum=0.9, epsilon=1e-5)(net)
    net = Activation('relu')(net)
    net = AveragePooling2D(8, 8)(net)
    net = Flatten()(net)
    net = Dense(10, activation='softmax')(net)
    return net

def cross_entropy(y_true, y_pred):
    cross_entropy = -tf.reduce_sum(y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)), axis=-1)
    return tf.reduce_mean(cross_entropy)

def l2_loss(model, weights=weight_decay):
    variable_list = []
    for v in model.trainable_variables:
        if 'kernel' or 'bias' in v.name:
            variable_list.append(tf.nn.l2_loss(v))
    return tf.add_n(variable_list) * weights

def accuracy(y_true, y_pred):
    correct_num = tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1))
    accuracy = tf.reduce_mean(tf.cast(correct_num, dtype=tf.float32))
    return accuracy

@tf.function
def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        prediction = model(x, training=True)
        ce = cross_entropy(y, prediction)
        l2 = l2_loss(model)
        loss = ce + l2
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return ce, prediction

@tf.function
def test_step(model, x, y):
    prediction = model(x, training=False)
    ce = cross_entropy(y, prediction)
    return ce, prediction

def train(model, optimizer, images, labels):
    sum_loss = 0
    sum_accuracy = 0

    # random shuffle
    seed = np.random.randint(0, 65536)
    np.random.seed(seed)
    np.random.shuffle(train_images)
    np.random.seed(seed)
    np.random.shuffle(train_labels)

    for i in tqdm(range(iterations_per_epoch)):
        x = images[i * batch_size: (i + 1) * batch_size, :, :, :]
        y = labels[i * batch_size: (i + 1) * batch_size, :]
        x = images_augment(x)

        loss, prediction = train_step(model, optimizer, x, y)
        sum_loss += loss
        sum_accuracy += accuracy(y, prediction)

    print('epoch:%d, ce_loss:%f, l2_loss:%f, accuracy:%f' %
          (epoch, sum_loss / iterations_per_epoch, l2_loss(model), sum_accuracy / iterations_per_epoch))

def test(model, images, labels):
    sum_loss = 0
    sum_accuracy = 0

    for i in tqdm(range(test_iterations)):
        x = images[i * test_batch_size: (i + 1) * test_batch_size, :, :, :]
        y = labels[i * test_batch_size: (i + 1) * test_batch_size, :]

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
    (train_images, train_labels, test_images, test_labels) = load_CIFAR('/home/user/Documents/dataset/Cifar-10')
    train_images, test_images = color_normalize(train_images, test_images)

    # get model
    img_input = Input(shape=(32, 32, 3))
    output = ResNet(img_input)
    model = models.Model(img_input, output)

    # show
    model.summary()

    # train
    learning_rate_schedules = optimizers.schedules.PiecewiseConstantDecay(boundaries, learning_rate)
    optimizer = optimizers.SGD(learning_rate=learning_rate_schedules, momentum=0.9, nesterov=True)

    for epoch in range(epoch_num):
        train(model, optimizer, train_images, train_labels)
        test(model, test_images, test_labels)

