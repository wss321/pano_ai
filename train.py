# -*- coding: utf-8 -*-
"""# 一、函数定义

## 1.导包
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import random
import tensorflow as tf
import keras
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard, EarlyStopping
from keras.losses import categorical_crossentropy
import seaborn as sns
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Model
from keras.models import model_from_json
from keras.layers.core import Dense, Activation
from keras import backend as K
from keras.engine import Layer
from keras.utils.generic_utils import get_custom_objects
from keras.applications.densenet import DenseNet121, preprocess_input
import os
from keras.layers import initializers, regularizers, constraints
from keras.engine import InputSpec
from sklearn.externals import joblib
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from keras.layers import PReLU
from sklearn.model_selection import KFold

os.environ['KERAS_BACKEND'] = 'tensorflow'

np.random.seed(0)
random.seed(0)
tf.set_random_seed(0)

"""## 2.数据处理

### (1).解析数据
"""
PRECESS_INPUT = False
USE_AMSoftmax = True
CLASS_BALANCE = False

MARGIN = 0.25
DenseNet_Model = 121
POOLING = 'avg'  # 'avg' 'max'
NUM_CHANNELS = 3
DATA_PATH = r'./train_data'
MODEL_DIR = './model'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
CNN_MODEL_JSON = '{}/model.json'.format(MODEL_DIR)
SVM_PATH = '{}/svc_clf.pkl'.format(MODEL_DIR)
# SAVE History
SAVE_HISTORY_FILE = os.path.join(MODEL_DIR, 'train_history.txt')

# 数据
ALL_SET = [str(i) for i in range(1, 11)]


def _pares_tf(example_proto):
    """解析函数"""
    features = {"id": tf.FixedLenFeature((), tf.int64),
                "data": tf.FixedLenFeature((256, 256), tf.float32),
                "label": tf.FixedLenFeature((), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    img_data = parsed_features["data"]
    #     max_ = tf.reduce_max(img_data)
    #     min_ = tf.reduce_min(img_data)
    #     standard_img = (img_data - min_) / (max_ - min_)
    return parsed_features["id"], tf.one_hot(parsed_features["label"] - 1, 5), img_data


def class_balance(x_data, y_data, title='Class balance:'):
    """每个类别数据均衡"""
    class_label_sum = np.sum(y_data, axis=0)
    print('{} training data original class number:{}\tdata length:{}'.format(title, class_label_sum, len(x_data)))
    max_len = max(class_label_sum)
    target_num = [max_len, max_len, max_len, max_len, max_len]
    x_new = []
    y_new = []
    for idx, i in enumerate(y_data):
        for k, j in enumerate(i):
            if j == 1.0:
                if class_label_sum[k] < target_num[k]:
                    x_new.append(x_data[idx])
                    y_new.append(y_data[idx])
                    class_label_sum[k] += 1
                break
    x_data = np.row_stack((x_data, x_new))
    y_data = np.row_stack((y_data, y_new))
    class_label_sum = np.sum(y_data, axis=0)
    print('Used class balence, final class number:{}\tdata length:{}'.format(class_label_sum, len(x_data)))
    return x_data, y_data


def get_data(file_path, shuffle_buffer_size=None, channels=3):
    for i in file_path:
        if not os.path.exists(i):
            raise IOError('{} not exist.'.format(i))
    pano = tf.data.TFRecordDataset([file_path])
    pano = pano.map(_pares_tf)
    if shuffle_buffer_size is not None:
        pano = pano.shuffle(buffer_size=shuffle_buffer_size)
    one_iterator = pano.make_one_shot_iterator().get_next()
    labels = []
    data = []
    with tf.Session() as sess:
        while True:
            try:
                img_id, one_hot_label, img = sess.run(one_iterator)
                labels.append(one_hot_label)
                img = np.asarray(img).reshape(256, 256, 1)
                if channels == 3:
                    img = img.repeat(3, axis=2)
                data.append(img)
            except:
                break
    data = np.asarray(data)
    labels = np.asarray(labels)
    return data, labels


def load_data(channels=3, shuffle_buffer_size=None):
    data_set = ALL_SET
    file_path = [os.path.join(DATA_PATH, 'TFcodeX_{}.tfrecord'.format(i)) for i in data_set]
    data, lables = get_data(file_path, shuffle_buffer_size=shuffle_buffer_size, channels=channels)
    if PRECESS_INPUT:
        data = preprocess_input(data, data_format='channels_last')
    return data, lables


def show_image(image, visual=True, title='', dpi=None):
    """可视化一张图片"""
    num_dims = len(list(np.shape(image)))
    if visual:
        plt.figure(dpi=dpi)
        plt.title(title)
        if num_dims == 2:
            plt.imshow(image, cmap='gray')
        elif num_dims == 3:
            plt.imshow(image)
        plt.show()
    return image


# # # (1).CNN模型


# # AMSoftmax

class AMSoftmax(Layer):
    def __init__(self, units, output_dim, margin,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs
                 ):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(AMSoftmax, self).__init__(**kwargs)
        self.units = units
        self.output_dim = output_dim
        self.margin = margin
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        inputs = tf.nn.l2_normalize(inputs, dim=-1)
        self.kernel = tf.nn.l2_normalize(self.kernel, dim=(0, 1))  # W归一化

        dis_cosin = K.dot(inputs, self.kernel)
        psi = dis_cosin - self.margin

        e_costheta = K.exp(self.output_dim * dis_cosin)
        e_psi = K.exp(self.output_dim * psi)
        sum_x = K.sum(e_costheta, axis=-1, keepdims=True)

        temp = e_psi - e_costheta
        temp = temp + sum_x

        output = e_psi / temp
        return output

    def get_config(self):
        config = {
            'units': self.units,
            'output_dim': self.output_dim,
            'margin': self.margin,

        }
        base_config = super(AMSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def amsoftmax_loss(y_true, y_pred):
    d1 = K.sum(y_true * y_pred, axis=-1)
    d1 = K.log(K.clip(d1, K.epsilon(), None))
    loss = -K.mean(d1, axis=-1)
    return loss


get_custom_objects().update({'AMSoftmax': AMSoftmax})

"""# 四、重训练预训练模型
## DenseNet
"""


def DenseNet(num_class=5, input_shape=(256, 256, 3), norm_rate=0.0,
             pooling='arg'):
    """重训练DenseNet"""
    base_model = DenseNet121(input_shape=input_shape, weights='imagenet', include_top=False, pooling=pooling)
    x = base_model.output
    x = Dense(1024, kernel_regularizer=regularizers.l2(norm_rate), name='fc1')(x)
    x = Activation('relu')(x)
    x = Dense(512, kernel_regularizer=regularizers.l2(norm_rate), name='fc2')(x)
    x = PReLU()(x)

    if USE_AMSoftmax:
        predictions = AMSoftmax(num_class, num_class, MARGIN)(x)
        model = Model(inputs=base_model.input, outputs=predictions, name='prediction')
        return model
    predictions = Dense(num_class, activation='softmax', name='prediction')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    return model


"""#### (a).训练CNN"""


# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def processing_img(img):
    def _add_random_noisy(img):
        rand = np.random.randint(100)
        if rand < 20:
            std = 3.0 / 255.0
            img = img + np.random.normal(0.0, std, img.shape)
            return img
        else:
            return img

    img = _add_random_noisy(img)
    rand = np.random.randint(100)
    if rand < 15:
        img = elastic_transform(img, img.shape[1] * 2, img.shape[1] * 0.08, img.shape[1] * 0.08)
    return img


def train_cnn(train_data, train_label, vali_data, vali_label, weight_file='weight.h5',
              train_batch_size=32, stop_patience=None, best_monitor='val_acc', loss_decay_patience=5,
              classifier_init_lr=1e-4, epochs=100):
    """训练一个CNN模型"""
    tf.reset_default_graph()
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    session = tf.Session(config=tfconfig)
    K.set_session(session)
    TB_LOG = './log/'
    tensorboard = TensorBoard(log_dir=TB_LOG)
    loss_decay = keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=loss_decay_patience, verbose=1)

    checkpoint = ModelCheckpoint(filepath=weight_file, monitor=best_monitor, verbose=1,
                                 mode='auto',
                                 save_best_only=True,
                                 save_weights_only=True)
    if stop_patience is None:
        callback_lists = [checkpoint, loss_decay, tensorboard]
    else:
        early_stopping = EarlyStopping(monitor='val_loss', patience=stop_patience, verbose=1, mode='auto')
        callback_lists = [early_stopping, checkpoint, loss_decay, tensorboard]
    # 加载数据
    print('training data shape:{}'.format(train_data.shape))
    print('validation data shape:{}'.format(vali_data.shape))
    print('Done.')
    model = DenseNet(num_class=5, input_shape=(256, 256, 3), norm_rate=1e-4, pooling=POOLING)
    optm = SGD(classifier_init_lr, momentum=0.9)
    if USE_AMSoftmax:
        loss_func = amsoftmax_loss
    else:
        loss_func = categorical_crossentropy

    model.compile(optimizer=optm, loss=loss_func, metrics=['accuracy'])

    model_json = model.to_json()
    json_file = CNN_MODEL_JSON
    with open(json_file, "w") as j_f:
        j_f.write(model_json)
    model.summary()
    train_len = len(train_data)
    keras_train_generator = ImageDataGenerator(featurewise_center=False,
                                               samplewise_center=False,
                                               featurewise_std_normalization=False,
                                               samplewise_std_normalization=False,
                                               zca_whitening=False,
                                               zca_epsilon=1e-6,
                                               rotation_range=40.,
                                               width_shift_range=0.2,
                                               height_shift_range=0.15,
                                               shear_range=0.2,
                                               zoom_range=0.2,
                                               channel_shift_range=0.,
                                               fill_mode='nearest',  # 'constant'
                                               cval=0.,
                                               brightness_range=None,
                                               horizontal_flip=True,
                                               vertical_flip=False,
                                               rescale=None,
                                               preprocessing_function=processing_img,
                                               data_format='channels_last')
    keras_train_generator.fit(train_data)
    h = model.fit_generator(
        generator=keras_train_generator.flow(train_data, train_label, batch_size=train_batch_size),
        verbose=1, steps_per_epoch=train_len / train_batch_size, epochs=epochs,
        callbacks=callback_lists,
        validation_data=(vali_data, vali_label))
    with open(SAVE_HISTORY_FILE, 'a') as f:
        f.write('{}:{}\n'.format('BEST VAL_ACC', max(h.history['val_acc'])))
        for key, value in h.history.items():
            f.write('{}:{}\n'.format(key, str(value)))
        f.write('\n')
    K.clear_session()
    return None


"""#### (b).融合预测"""


# # 1.可视化

def _visual_err_img_and_heatmap(argmax_predict_labels, argmax_true_labels, corr_img_data, vis=True, title=''):
    """可视化错误数据(图片)和heat map"""
    errors = np.where(~np.equal(argmax_predict_labels, argmax_true_labels))[0].ravel()
    predict_acc = (len(argmax_true_labels) - len(errors)) / (len(argmax_true_labels) * 1.0)
    with open(SAVE_HISTORY_FILE, 'a') as f:
        f.write('CNN predict {} acc:{}\n'.format(title, predict_acc))
    print('CNN predict {} acc:{}\n'.format(title, predict_acc))
    label = ['1-rose', '2-sunflower', '3-daisy', '4-dandelion', '5-tulips']
    if vis:
        for i in errors:
            true_label = argmax_true_labels[i] + 1
            false_label = argmax_predict_labels[i] + 1
            show_image(corr_img_data[i].reshape(256, 256, 3),
                       title='true:{} ~ false:{}'.format(label[true_label - 1], label[false_label - 1]))
        heat = np.zeros(shape=(5, 5))
        for i, j in zip(argmax_true_labels, argmax_predict_labels):
            heat[-i + 4][j] += 1
        fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
        df = pd.DataFrame(heat, index=label[::-1], columns=label)
        sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True, square=True, cmap="YlGnBu")
        ax.set_title('{} Prediction Accuracy:{:.4f}'.format(title, predict_acc))
        ax.set_ylabel('true_label')
        ax.set_xlabel('predict_label')


"""## 2.预测"""


def get_acc(pred_label, true_label):
    errors = np.where(~np.equal(pred_label, true_label))[0].ravel()
    predict_acc = (len(true_label) - len(errors)) / (len(true_label) * 1.0)
    return predict_acc


def _cnn_predict(model_json_file, model_weight_file, data, verbose=1):
    """数据通过CNN进行预测"""
    print('loading model {}'.format(model_json_file))
    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)
    K.set_session(session)
    model = model_from_json_and_weight(model_json_file, model_weight_file)
    predict = model.predict(data, verbose=verbose)
    K.clear_session()
    return predict


def _cnn_merge_predict(img_data, one_hot_label, model_json, model_weight, model_path='./', best_only=True):
    """数据通过多个CNN进行预测"""
    original_predict_label = []
    predict = _cnn_predict(model_json, model_weight, img_data, verbose=1)
    original_predict_label.append(predict)
    original_predict_label = np.asarray(original_predict_label)
    return original_predict_label, one_hot_label


def _cnn_mean_error(img_data, one_hot_label, model_json, model_weight, best_only=True, vis=False):
    """数据多个CNN平均后的误差"""
    original_predict_label, one_hot_label = _cnn_merge_predict(img_data, one_hot_label, model_json, model_weight,
                                                               best_only=best_only)
    argmax_true_label = np.argmax(one_hot_label, axis=1)
    # 平均预测
    mean_predict = np.mean(original_predict_label, axis=0)
    print(mean_predict)
    argmax_predict_label = np.argmax(mean_predict, axis=1)
    _visual_err_img_and_heatmap(argmax_predict_label, argmax_true_label, img_data, vis=vis, title='test-mean')
    return original_predict_label, argmax_true_label


def cnn_test(test_data, test_label, model_json, model_weight, best_only=True, vis=False):
    """预测和选择可视化CNN测试集分类结果"""
    predicts, ground_true_label = _cnn_mean_error(test_data, test_label, model_json, model_weight, best_only=best_only,
                                                  vis=vis)
    for idx, predict in enumerate(predicts):
        predict_label = np.argmax(predict, axis=1)
        _visual_err_img_and_heatmap(predict_label, ground_true_label, test_data, vis=vis,
                                    title='test-{}'.format(idx))
    return None


def model_from_json_and_weight(model_json_file, model_weight_file=None):
    # load json and create model
    print('Loading model from json:{} ...'.format(model_json_file))
    json_file = open(model_json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    if USE_AMSoftmax:
        model = model_from_json(loaded_model_json, custom_objects={'amsoftmax_loss': amsoftmax_loss})
    else:
        model = model_from_json(loaded_model_json)
    # load weights into new model
    print('Done.')
    if model_weight_file is not None:
        print('Loading model weight from:{} ...'.format(model_weight_file))
        model.load_weights(model_weight_file)
        print('Done.')
    return model


"""# 二、运行主函数"""


def get_cnn_feature(model_json_file, model_weight_file, data, summary=False):
    """获取数据通过CNN后的属性"""
    print('data shape', data.shape)
    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)
    K.set_session(session)
    model = model_from_json_and_weight(model_json_file, model_weight_file)
    top_layer = 'fc1'
    model = Model(inputs=model.input, outputs=model.get_layer(top_layer).output)
    if summary:
        model.summary()
    feature = model.predict(data, verbose=1)
    K.clear_session()
    return feature


def get_feature(img_data, one_hot_label, model_json_file, model_weight_file):
    """获取数据通过多个CNN平均后的属性"""
    predicts = get_cnn_feature(model_json_file, model_weight_file, data=img_data, summary=False)
    ground_true_argmax_label = np.argmax(one_hot_label, axis=1)
    return predicts, ground_true_argmax_label


"""#### (b).训练svm"""


# 训练
def train_svm_classifier(features, labels, search=False, save_to='svc_clf.pkl', test_size=0.2):
    """训练SVM"""
    # prepare training and test datasets
    x_train, y_train = features, labels
    if search:

        # train and then classify the images with C-Support Vector Classification
        param = [
            {
                "kernel": ["linear"],
                "C": [1, 10, 100, 1000]
            },
            {
                "kernel": ["rbf"],
                "C": [1, 10, 100, 1000],
                "gamma": [1e-2, 1e-3, 1e-4, 1e-5]
            }
        ]

        print('C-Support Vector Classification starting ...')
        start_time = time.time()
        svm_c = SVC(probability=True)
        clf = model_selection.GridSearchCV(svm_c, param, cv=10, n_jobs=4, verbose=3)

        clf.fit(x_train, y_train)
        print("\nBest parameters set:")
        print(clf.best_params_)
        with open(SAVE_HISTORY_FILE, 'a') as f:
            f.write('SVM Best parameters set:{}\nSearch elaspe {} seconds\n'.format(clf.best_params_,
                                                                                    time.time() - start_time))
        print("%f seconds" % (time.time() - start_time))
    else:
        clf = SVC(kernel='rbf', C=1, gamma=0.001)
        clf.fit(x_train, y_train)
    joblib.dump(clf, save_to)
    y_pred = clf.predict(x_train)

    # confusion matrix computation and display
    predict_acc = accuracy_score(y_train, y_pred) * 100
    print("CNN-C-SVC train Accuracy: {0:0.4f}".format(predict_acc))
    with open(SAVE_HISTORY_FILE, 'a') as f:
        f.write('SVM train Accuracy:{}\n'.format(predict_acc))
    return None


def svm_predict_and_visual(svm_model_path, feature, corr_img_data, argmax_labels, vis=True):
    """用SVM模型预测，并可以进行可视化"""
    clf = joblib.load(svm_model_path)
    predict = clf.predict(feature)
    acc = accuracy_score(predict, argmax_labels)
    # print('SVM Accuracy:{}'.format(acc))
    if vis:
        _visual_err_img_and_heatmap(predict, argmax_labels, corr_img_data)
    return predict + 1, acc


def train_svm(svm_train_data, svm_train_label, svm_test_data, svm_test_label, model_json_file, model_weight_file,
              search=False, save_to='svm_clf.pkl',
              test=False, vis=False):
    """训练SVM--1"""
    # 1. 特征提取
    print('SVM training data:{}'.format(svm_train_data.shape))
    train_feature_data, train_true_labels = get_feature(svm_train_data, svm_train_label, model_json_file,
                                                        model_weight_file
                                                        )
    print(train_feature_data.shape)
    # 2.训练SVM
    train_svm_classifier(train_feature_data, train_true_labels, search=search, save_to=save_to)

    if test:
        test_feature_data, test_true_labels = get_feature(svm_test_data, svm_test_label, model_json_file,
                                                          model_weight_file)
        print(svm_test_data.shape)
        test_predict, test_acc = svm_predict_and_visual(save_to, test_feature_data, svm_test_data,
                                                        test_true_labels,
                                                        vis=vis)
        print('Test prediction :\n{}'.format(test_predict))
        print('Test Accuracy:{}'.format(test_acc))
        with open(SAVE_HISTORY_FILE, 'a') as f:
            f.write('SVM Test Accuracy:{}\n'.format(test_acc))
    return None


"""# 二、运行主函数"""


def main():
    # 加载数据
    print('Loading all data ...')
    all_data, all_label = load_data(shuffle_buffer_size=None)
    kfold = KFold(n_splits=7, shuffle=True, random_state=0)
    i = 0
    kfold_generator = kfold.split(all_data, all_label)
    for train_index, test_index in kfold_generator:
        i += 1
        weight_file = '{}/weight{}.h5'.format(MODEL_DIR, i)
        train_data = all_data[train_index]
        train_label = all_label[train_index]
        test_data = all_data[test_index]
        test_label = all_label[test_index]
        # 训练CNN
        train_cnn(train_data, train_label, test_data, test_label, weight_file,
                  train_batch_size=32,
                  stop_patience=15,
                  best_monitor='val_loss',
                  loss_decay_patience=3,
                  classifier_init_lr=1e-3,
                  epochs=1000)
        cnn_test(test_data, test_label, CNN_MODEL_JSON, weight_file, best_only=True, vis=False)
        train_svm(train_data, train_label, test_data, test_label, CNN_MODEL_JSON, weight_file, test=True, vis=False,
                  save_to='{}/svm_clf{}.pkl'.format(MODEL_DIR, i))

    return None


if __name__ == '__main__':
    main()
