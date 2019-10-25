import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, GlobalAvgPool2D, BatchNormalization, Flatten, Dense

# network confing
block_type = {18: 'basic block',
              34: 'basic block',
              50: 'bottlenect block',
              101: 'bottlenect block',
              152: 'bottlenect block'}

block_num = {18: (2, 2, 2, 2),
             34: (3, 4, 6, 3),
             50: (3, 4, 6, 3),
             101: (3, 4, 23, 3),
             152: (3, 4, 36, 3)}

filter_num = (64, 128, 256, 512)

class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=(1, 1), **kwargs):
        self.strides = strides
        if strides != (1, 1):
            self.shortcut = Conv2D(filters, (1, 1), name='projection', strides=strides, padding='same')
            self.shortcut_bn = BatchNormalization(name='shortcut_bn', momentum=0.9, epsilon=1e-5)

        self.conv_0 = Conv2D(filters, (3, 3), name='conv_0', strides=strides, padding='same')
        self.conv_1 = Conv2D(filters, (3, 3), name='conv_1', padding='same')
        self.bn_0 = BatchNormalization(name='bn_0', momentum=0.9, epsilon=1e-5)
        self.bn_1 = BatchNormalization(name='bn_1', momentum=0.9, epsilon=1e-5)

        super(BasicBlock, self).__init__(**kwargs)

    def call(self, inputs, training):
        net = self.conv_0(inputs)
        net = self.bn_0(net, training=training)
        net = tf.nn.relu(net)

        net = self.conv_1(net)
        net = self.bn_1(net, training=training)
        net = tf.nn.relu(net)

        shortcut = inputs
        if self.strides != (1, 1):
            shortcut = self.shortcut(shortcut)
            shortcut = self.shortcut_bn(shortcut)

        net = net + shortcut
        net = tf.nn.relu(net)
        return net

class BottleneckBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=(1, 1), projection=False, **kwargs):
        self.strides = strides
        self.projection = projection
        if projection or strides != (1, 1):
            self.shortcut = Conv2D(filters * 4, (1, 1), name='projection', strides=strides, padding='same')
            self.shortcut_bn = BatchNormalization(name='shortcut_bn', momentum=0.9, epsilon=1e-5)

        self.conv_0 = Conv2D(filters, (1, 1), name='conv_0', strides=strides, padding='same')
        self.conv_1 = Conv2D(filters, (3, 3), name='conv_1', padding='same')
        self.conv_2 = Conv2D(filters * 4, (1, 1), name='conv_2', padding='same')
        self.bn_0 = BatchNormalization(name='bn_0', momentum=0.9, epsilon=1e-5)
        self.bn_1 = BatchNormalization(name='bn_1', momentum=0.9, epsilon=1e-5)
        self.bn_2 = BatchNormalization(name='bn_2', momentum=0.9, epsilon=1e-5)

        super(BottleneckBlock, self).__init__(**kwargs)

    def call(self, inputs, training):
        net = self.conv_0(inputs)
        net = self.bn_0(net, training=training)
        net = tf.nn.relu(net)

        net = self.conv_1(net)
        net = self.bn_1(net, training=training)
        net = tf.nn.relu(net)

        net = self.conv_2(net)
        net = self.bn_2(net, training=training)

        shortcut = inputs
        if self.projection or self.strides != (1, 1):
            shortcut = self.shortcut(shortcut)
            shortcut = self.shortcut_bn(shortcut)

        net = net + shortcut
        net = tf.nn.relu(net)
        return net


class ResNet(models.Model):
    def __init__(self, layer_num, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        if block_type[layer_num] == 'basic block':
            self.block = BasicBlock
        else:
            self.block = BottleneckBlock

        self.conv0 = Conv2D(64, (7, 7), strides=(2, 2), name='conv0', padding='same')
        self.bn = BatchNormalization(name='bn', momentum=0.9, epsilon=1e-5)

        self.block_collector = []
        for layer_index, (b, f) in enumerate(zip(block_num[layer_num], filter_num), start=1):
            if layer_index == 1:
                if block_type[layer_num] == 'basic block':
                    self.block_collector.append(self.block(f, name='conv1_0'))
                else:
                    self.block_collector.append(self.block(f, projection=True, name='conv1_0'))
                for block_index in range(1, b):
                    self.block_collector.append(self.block(f, name='conv1_{}'.format(block_index)))
            else:
                self.block_collector.append(self.block(f, strides=(2, 2), name='conv{}_{}'.format(layer_index, 0)))
                for block_index in range(1, b):
                    self.block_collector.append(self.block(f, name='conv{}_{}'.format(layer_index, block_index)))

        self.global_average_pooling = GlobalAvgPool2D()
        self.fc = Dense(1000, name='fully_connected', activation='softmax')

    def call(self, inputs, training):
        net = self.conv0(inputs)
        net = self.bn(net)
        net = tf.nn.relu(net)
        print('input', inputs.shape)
        print('conv0', net.shape)
        net = tf.nn.max_pool2d(net, ksize=(3, 3), strides=(2, 2), padding='SAME')
        print('max-pooling', net.shape)

        for block in self.block_collector:
            net = block(net, training)
            print(block.name, net.shape)

        net = self.global_average_pooling(net)
        print('global average-pooling', net.shape)
        net = self.fc(net)
        print('fully connected', net.shape)
        return net

if __name__ == '__main__':
    model = ResNet(152)
    model.build(input_shape=(None, 224, 224, 3))