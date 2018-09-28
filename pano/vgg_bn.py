from keras import Model
from keras.layers import Flatten, Dense, Input, Convolution2D, MaxPooling2D, BatchNormalization, Activation, \
    GlobalAveragePooling2D
from keras import regularizers

# size of pooling area for max pooling
_pool_size = (2, 2)
_stride = (2, 2)
# convolution kernel size
_kernel_size = (3, 3)

_filters = [16, 32, 32, 64, 128]
_layer_num = [1, 1, 2, 2, 3]


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


def VGG_BN(num_class, filters, layer_num, input_shape=(256, 256, 1,), norm_rate=0.0, name=None):
    """
    VGG model with batch-normalize on each block.
    :param num_class: the number of your DataSet classes
    :param filters: a list of filters
    :param layer_num: a list of the number of each block
    :param name: default None. it can be given list to name the block
    :param input_shape: the shape of input
    :param norm_rate: l2 norm rate
    :return: vgg batch-normalize model
    """
    if len(filters) < len(layer_num):
        print('ERROR:filters length must be equal or longer than layer_num.')
    if name is None:
        name = ['Conv{}'.format(i + 1) for i in range(len(layer_num))]
    inputs = Input(shape=input_shape, name='input')
    x = inputs
    for i in range(len(layer_num)):
        x = _block(layer_num=layer_num[i], inputs=x, filters=filters[i], name=name[i])
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, kernel_regularizer=regularizers.l2(norm_rate))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dense(512, kernel_regularizer=regularizers.l2(norm_rate))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(num_class, activation='softmax', name='prediction',
                        kernel_regularizer=regularizers.l2(norm_rate))(x)
    model = Model(inputs=inputs, outputs=predictions)
    return model


if __name__ == '__main__':
    from keras.optimizers import Adam

    model = VGG_BN(205, filters=[16, 32, 32, 64, 128], layer_num=[1, 1, 2, 2, 3], norm_rate=0.0)
    print("DONE.")
    optimizer = Adam(1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])  #
    model.summary()
