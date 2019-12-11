import tensorflow as tf
import numpy as np

def add_noise(x, y):
    x += np.random.uniform(0.0, 0.01)
    return x, y

def one_hot(x, y):
    if y == 0:
        return x, np.array([1, 0])
    else:
        return x, np.array([0, 1])

if __name__ == '__main__':
    data = np.array([0.1, 0.4, 0.6, 0.2, 0.8, 0.8, 0.4, 0.9, 0.3, 0.2])
    print(data)
    label = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 0])
    print(label)

    dataset = tf.data.Dataset.from_tensor_slices((data, label))

    print('traversal all data')
    for x, y in dataset:
        print(x, y)

    print('use iterator')
    dataset = dataset.repeat()
    it = dataset.__iter__()
    for i in range(20):
        x, y = it.next()
        print(x, y)

    print('random shuffle')
    dataset = dataset.shuffle(buffer_size=10)
    it = dataset.__iter__()
    for i in range(10):
        x, y = it.next()
        print(x, y)

    print('batch')
    dataset_batch = dataset.batch(batch_size=5)
    it = dataset_batch.__iter__()
    for i in range(2):
        x, y = it.next()
        print(x, y)

    print('one hot')
    dataset_one_hot = dataset.map(one_hot)
    it = dataset_one_hot.__iter__()
    for i in range(10):
        x, y = it.next()
        print(x, y)

    print('add noise')
    dataset_add_noise = dataset.map(add_noise)
    it = dataset_add_noise.__iter__()
    for i in range(10):
        x, y = it.next()
        print(x, y)

    print('add noise')
    dataset_add_noise = dataset.map(lambda x, y: tf.py_function(add_noise, inp=[x, y], Tout=[tf.float64, tf.int64]),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    it = dataset_add_noise.__iter__()
    for i in range(10):
        x, y = it.next()
        print(x, y)
