from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from densenet import DenseNetImageNet121
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Activation,Flatten
from keras import backend as K


def fine_tune_inceptionV3(num_class=5, input_shape=(256, 256, 3)):
    # create the base pre-trained model
    base_model = InceptionV3(input_shape=input_shape, weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(256, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(num_class, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def fine_tune_VGG19(num_class=5, input_shape=(256, 256, 3)):
    # create the base pre-trained model
    base_model = VGG19(input_shape=input_shape, weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    # x = Flatten()(x)

    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(num_class, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def fine_tune_densenet121(num_class=5, input_shape=(256, 256, 3)):
    base_model = DenseNetImageNet121(input_shape=input_shape, weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False

    # add a global spatial average pooling layer
    x = base_model.output
    x = BatchNormalization()(x)
    # x = Flatten()(x)

    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(num_class, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model


if __name__ == '__main__':
    from keras.optimizers import Adam

    # model = fine_tune_inceptionV3()
    model = fine_tune_densenet121()
    optimizer = Adam(1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])  #
    model.summary()
