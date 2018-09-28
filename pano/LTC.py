from keras import Model
from keras.layers import Flatten, Dense, Input, Convolution2D, MaxPooling2D, BatchNormalization, Activation, \
    GlobalAveragePooling2D, AveragePooling2D, Reshape
from keras import regularizers
import numpy as np

# size of pooling area for max pooling
_pool_size = (2, 2)
_stride = (2, 2)
# convolution kernel size
_kernel_size = (3, 3)
BATCH_NUM_PER_CLASS = 15
FEATURE_DIM = 128


def _conv_act_bn(inputs, filters, name, kernel_size=(3, 3), norm_rate=0.0, padding='same'):
    x = Convolution2D(filters, kernel_size=kernel_size, padding=padding, name=name,
                      kernel_regularizer=regularizers.l2(norm_rate))(inputs)

    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    return x


def _block(layer_num, inputs, filters, name, kernel_size=(3, 3), norm_rate=0.0, padding='same'):
    x = inputs

    for i in range(layer_num):
        x = _conv_act_bn(x, filters, name=name + '_{}'.format(i + 1), kernel_size=kernel_size, norm_rate=norm_rate,
                         padding=padding)
    x = MaxPooling2D(pool_size=_pool_size, strides=(2, 2))(x)
    return x


# def LTC_BN(num_class, filters, layer_num, input_shape=(256, 256, 1,), norm_rate=0.0, name=None):
#     """
#     VGG model with batch-normalize on each block.
#     :param num_class: the number of your DataSet classes
#     :param filters: a list of filters
#     :param layer_num: a list of the number of each block
#     :param name: default None. it can be given list to name the block
#     :param input_shape: the shape of input
#     :param norm_rate: l2 norm rate
#     :return: vgg batch-normalize model
#     """
#     if len(filters) < len(layer_num):
#         print('ERROR:filters length must be equal or longer than layer_num.')
#     if name is None:
#         name = ['Conv{}'.format(i + 1) for i in range(len(layer_num))]
#     inputs = Input(shape=input_shape, name='input')
#     x = inputs
#     for i in range(len(layer_num)):
#         x = _block(layer_num=layer_num[i], inputs=x, filters=filters[i], name=name[i])
#     x = AveragePooling2D()(x)
#     x = Flatten()(x)
#     x = Dense(1024, kernel_regularizer=regularizers.l2(norm_rate))(x)
#     x = Activation('relu')(x)
#     x = BatchNormalization()(x)
#     x = Dense(512, kernel_regularizer=regularizers.l2(norm_rate))(x)
#     x = Activation('relu')(x)
#     x = BatchNormalization()(x)
#     predictions = Dense(num_class, activation='softmax', name='prediction',
#                         kernel_regularizer=regularizers.l2(norm_rate))(x)
#     model = Model(inputs=inputs, outputs=predictions)
#     return model
#
#
# def CNNEncoder(input_shape, norm_rate=0.0):
#     """docstring for ClassName"""
#     inputs = Input(shape=input_shape, name='input')
#     x = Dense(1024, kernel_regularizer=regularizers.l2(norm_rate))(inputs)
#     x = Activation('relu')(x)
#     x = BatchNormalization()(x)
#     x = Dense(512, kernel_regularizer=regularizers.l2(norm_rate))(x)
#     x = Activation('relu')(x)
#     x = BatchNormalization()(x)
#     return x
def get_concentrate_rp(base_model, train_data, sample_data, train_label):
    """
    提取sample集和train集特征并结合
    :param base_model:
    :param sample_data: [-1,n,n,f] 要求每类的包含的图片数相同，且同类紧挨着,从第一类到第五类
    :param train_data: 训练集数据
    :param train_label:训练集标签
    :return: 联合特征及训练标签
    """
    sample_features = base_model.predict(sample_data, verbose=1)
    sample_data = None
    train_features = base_model.predict(train_data, verbose=1)
    train_data = None
    # print(sample_features.shape)  # (100, 8, 8, 512)
    # print(train_features.shape)  # (220, 8, 8, 512)
    sample_features = np.reshape(sample_features, newshape=(
        5, 20, base_model.output.shape[1], base_model.output.shape[2], base_model.output.shape[3]))
    # print(sample_features.shape)  # (5, 20, 8, 8, 512)
    sample_features = np.sum(sample_features, 1)
    # print(sample_features.shape)  # (5, 8, 8, 512)
    sample_features = np.expand_dims(sample_features, axis=0).repeat(train_features.shape[0], axis=0)
    # print(sample_features.shape)  # (220, 5, 8, 8, 512)
    train_features = np.expand_dims(train_features, axis=0).repeat(5, axis=0)
    # print(train_features.shape)  # (5, 220, 8, 8, 512)
    train_features = np.transpose(train_features, (1, 0, 2, 3, 4))
    # print(train_features.shape)  # (220, 5, 8, 8, 512)
    relation_pairs = np.concatenate((sample_features, train_features), axis=4).reshape(train_features.shape[0], -1,
                                                                                       train_features.shape[2],
                                                                                       train_features.shape[2])
    # print(relation_pairs.shape)  # (220*5, 512*2, 5, 5)==(1100, 1024, 8, 8)
    ralation_label = train_label
    # print(relation_label.shape)  # (220, 5)
    return relation_pairs, ralation_label


def LTC_BN(num_class, input_shape, norm_rate=0.0):
    """docstring for RelationNetwork"""

    inputs = Input(shape=input_shape, name='input')

    x = _conv_act_bn(inputs, filters=16, name='RN_Conv1')
    x = _conv_act_bn(x, filters=32, name='RN_Conv2')
    x = Flatten()(x)
    x = Dense(1024, kernel_regularizer=regularizers.l2(norm_rate))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    # x = Reshape(target_shape=(-1,5))(x)
    x = Dense(num_class, kernel_regularizer=regularizers.l2(norm_rate))(x)
    predictions = Activation('sigmoid')(x)
    # = Reshape(target_shape=(220, -1))(x)
    # predictions =

    # predictions = Dense(num_class, activation='sigmoid', name='prediction',
    #                     kernel_regularizer=regularizers.l2(norm_rate))(x)
    model = Model(inputs=inputs, outputs=predictions)
    return model


if __name__ == '__main__':
    from keras.optimizers import Adam

    model = LTC_BN(5, input_shape=(5, 5, 128), norm_rate=0.0)
    optimizer = Adam(1e-4)
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])  #
    model.summary()
