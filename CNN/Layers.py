import os
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import BatchNormalization, Flatten, Dense, Input, AveragePooling2D, Activation
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class Conv2D(tf.keras.layers.Layer):
    def __init__(self, output_dim, kernel_size=(3, 3), strides=(1, 1, 1, 1), **kwargs):
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.strides = strides
        super(Conv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        kernel_shape = tf.TensorShape((self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.output_dim))
        self.kernel = self.add_weight(name='kernel',
                                      shape=kernel_shape,
                                      initializer=tf.initializers.he_normal())

    def call(self, inputs):
        output = tf.nn.conv2d(inputs, filters=self.kernel, strides=self.strides, padding='SAME')
        return output

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, output_dim, strides=(1, 1, 1, 1), **kwargs):
        self.strides = strides
        if strides != (1, 1, 1, 1):
            self.shortcut = Conv2D(output_dim, kernel_size=(1, 1), strides=self.strides)
        self.conv_0 = Conv2D(output_dim, strides=self.strides)
        self.conv_1 = Conv2D(output_dim)
        self.bn_0 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.bn_1 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        super(ResBlock, self).__init__(**kwargs)

    def call(self, inputs, training):
        net = self.bn_0(inputs, training=training)
        net = tf.nn.relu(net)

        if self.strides != (1, 1, 1, 1):
            shortcut = self.shortcut(net)
        else:
            shortcut = inputs

        net = self.conv_0(net)
        net = self.bn_1(net, training=training)
        net = tf.nn.relu(net)
        net = self.conv_1(net)

        output = net + shortcut
        return output

def ResNet(inputs):
    net = Conv2D(16)(inputs)

    for i in range(stack_n):
        net = ResBlock(16)(net)

    net = ResBlock(32, strides=(1, 2, 2, 1))(net)
    for i in range(stack_n - 1):
        net = ResBlock(32)(net)

    net = ResBlock(64, strides=(1, 2, 2, 1))(net)
    for i in range(stack_n - 1):
        net = ResBlock(64)(net)

    net = BatchNormalization(momentum=0.9, epsilon=1e-5)(net)
    net = Activation('relu')(net)
    net = AveragePooling2D(8, 8)(net)
    net = Flatten()(net)
    net = Dense(10, activation='softmax')(net)
    return net

if __name__ == '__main__':
    # gpu config
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

    # get model
    stack_n = 3
    img_input = Input(shape=(32, 32, 3))
    output = ResNet(img_input)
    model = models.Model(img_input, output)

    # show
    model.summary()
