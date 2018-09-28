from dataset import load_train_data_3C, load_test_data_3C, load_test_data_1C, load_train_data_1C
import random
import os
import tensorflow as tf
from keras import backend as K
import keras
from keras.models import load_model
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard, EarlyStopping
from vgg_bn import VGG_BN
from data_generator import DataGenerator, distorted_batch, NUM_CHANNELS
from densenet import DenseNet

random.seed(0)
tf.set_random_seed(0)
if __name__ == '__main__':
    TRAINING_DIR = 'output'
    MODEL_FOLDER = r'{}/model/'.format(TRAINING_DIR)  # 模型保存地址
    TB_LOG = '{}/log/'.format(TRAINING_DIR)  # tensorbord 文件地址
    CKPT_PATH = '{}/checkoutpoint'.format(TRAINING_DIR)  # 查看点路径
    BEST_CLASSIFY_CKPT_FILE = '{}/best_one.ckpt'.format(CKPT_PATH)
    MODEL_DIR = '{}/dnn_classifier.h5'.format(MODEL_FOLDER)
    OPTIMIZER = 'adam'

    classifier_init_lr = 1e-4
    vgg_norm_rate = 0.0
    IMAGE_SIZE = 256
    resize = 256
    train_batch_size = 32

    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)
    if not os.path.exists(TB_LOG):
        os.makedirs(TB_LOG)
    if not os.path.exists(CKPT_PATH):
        os.makedirs(CKPT_PATH)

    # model = VGG16(input_shape=(256, 256, 1), classes=5)

    tf.reset_default_graph()
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    session = tf.Session(config=tfconfig)
    K.set_session(session)

    # 回调函数
    tensorboard = TensorBoard(log_dir=TB_LOG)
    checkpoint = ModelCheckpoint(filepath=BEST_CLASSIFY_CKPT_FILE, monitor='val_acc', mode='auto',
                                 save_best_only='True')
    losscalback = keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=1, verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    callback_lists = [earlystop, checkpoint, losscalback, tensorboard]

    if OPTIMIZER == 'adam':
        optm = Adam(classifier_init_lr)
    else:
        optm = SGD(lr=classifier_init_lr)

    model = VGG_BN(num_class=5, input_shape=(256, 256, NUM_CHANNELS), filters=[16, 32, 32, 64, 64],
                   layer_num=[1, 1, 2, 2, 3],
                   norm_rate=vgg_norm_rate)
    # model = DenseNet((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), depth=64, nb_dense_block=4,
    #                  growth_rate=12, bottleneck=True, dropout_rate=0.2, reduction=0.0,
    #                  classes=5)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])  #
    model.summary()

    # 加载数据
    print('Loading data....')
    if NUM_CHANNELS==3:
        test_data, test_label = load_test_data_3C()
        train_data, train_label = load_train_data_3C()
    else:
        test_data, test_label = load_test_data_1C()
        train_data, train_label = load_train_data_1C()
    # test_data = session.run(tf.image.resize_images(test_data, [IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS]))
    # if NUM_CHANNELS == 3:
    #     for i in test_data_temp:
    #         p = [i, i, i]
    print(train_data.shape)
    print(test_data.shape)

    print('Done.')
    x = K.placeholder(dtype=tf.float32, shape=(train_batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    distort_op = distorted_batch(x, IMAGE_SIZE, resize)
    train_generator = DataGenerator(session, distort_op, x, train_data, train_label, batch_size=train_batch_size,
                                    distort=True,
                                    shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_data = None
    # 训练
    print('Training......')
    h = model.fit_generator(generator=train_generator, verbose=1,
                            epochs=30, callbacks=callback_lists, validation_data=(test_data, test_label))
    # h = model.fit(x=train_data, y=train_label, epochs=30, validation_data=(test_data, test_label))
    # callback_lists=callback_lists)
    with open(os.path.join(TRAINING_DIR, 'train_history,txt'), 'a') as f:
        for key, value in h.history.items():
            f.write('{}:{}\n'.format(key, str(value)))
    model.save(MODEL_DIR)
