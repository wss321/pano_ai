from keras import Model
from keras.layers import Flatten, Dense, Input, Convolution2D, MaxPooling2D, BatchNormalization, Activation
from keras import regularizers

# size of pooling area for max pooling
_pool_size = (2, 2)
_stride = (2, 2)
# convolution kernel size
_kernel_size = (3, 3)

_filters = [16, 32, 32, 64, 128, 128]
_layer_num = [1, 1, 2, 2, 3, 3]
_name = ['Conv{}'.format(i+1) for i in range(6)]


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


def VGG_BN(num_class, input_shape=(256, 256, 1,), norm_rate=0.0):
    inputs = Input(shape=input_shape, name='input')
    x = inputs
    for i in range(6):
        x = _block(layer_num=_layer_num[i], inputs=x, filters=_filters[i], name=_name[i])
    x = Flatten()(x)

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

    model = VGG_BN(205, norm_rate=0.0)
    print("DONE.")
    optimizer = Adam(1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])  #
    model.summary()
