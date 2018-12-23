# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import os
import re
from train import model_from_json_and_weight, CNN_MODEL_JSON, PRECESS_INPUT, preprocess_input, MODEL_DIR, \
    get_cnn_feature, get_data, joblib

TEST_DATA_DIR = './test_data'
USE_SVM = True


def encode_one_hot(data, classes=None):
    """one hot 编码"""
    if not classes:
        classes = max(data) + 1
    data = np.asarray(data)
    return np.asarray(np.arange(classes) == data[:, None]).astype(np.integer).tolist()


def decode_one_hot(data):
    """one hot 解码"""
    data = np.asarray(data)
    shape = data.shape
    if len(shape) == 1:
        axis = 0
    else:
        axis = 1
    decode = np.argmax(data, axis=axis)
    return decode


def load_data(filename, channels=3, shuffle_buffer_size=None):
    data_set = list()
    data_set.append(filename)
    data, lables = get_data(data_set, shuffle_buffer_size=shuffle_buffer_size, channels=channels)
    if PRECESS_INPUT:
        data = preprocess_input(data, data_format='channels_last')
    return data, lables


def find_weight(model_dir):
    files = os.listdir(model_dir)
    weights = []
    for file in files:
        if 'weight' in file:
            file = os.path.join(model_dir, file)
            weights.append(file)
    weights = sorted(weights)
    return weights


def find_svm_model(model_dir):
    files = os.listdir(model_dir)
    svm_models = []
    for file in files:
        if 'svm' in file:
            file = '{}/{}'.format(model_dir, file)
            svm_models.append(file)
    svm_models = sorted(svm_models)
    return svm_models


# def get_acc(pred_label, true_label):
#     errors = np.where(~np.equal(pred_label, true_label))[0].ravel()
#     predict_acc = (len(true_label) - len(errors)) / (len(true_label) * 1.0)
#     return predict_acc


def get_number(str):
    num = re.findall(r"\d+\.?\d*", str)[0]
    num = float(num)
    return num


def model_test(filename):
    test_data, _ = load_data('{}/{}'.format(TEST_DATA_DIR, filename), channels=3, shuffle_buffer_size=None)
    # true_label = np.argmax(test_label, axis=1)
    if PRECESS_INPUT:
        test_data = preprocess_input(test_data)
    weights = find_weight(MODEL_DIR)
    cnn_predictions = []
    svm_predictions = []
    # CNN 预测
    svm_models = find_svm_model(MODEL_DIR)
    for weight_file in weights:
        model = model_from_json_and_weight(CNN_MODEL_JSON, weight_file)
        pred = model.predict(test_data, verbose=1)
        pred_labels = np.argmax(pred, axis=1)
        cnn_predictions.append(pred_labels)
        # print('cnn acc:{}'.format(get_acc(pred_labels, true_label)))
        if USE_SVM:
            for svm_model_file in svm_models:
                if get_number(svm_model_file) == get_number(weight_file):
                    # 提取特征
                    clf = joblib.load(svm_model_file)
                    feature = get_cnn_feature(CNN_MODEL_JSON, weight_file, test_data)
                    predict = clf.predict(feature)
                    svm_predictions.append(predict)

                    # print('svm acc:{}'.format(get_acc(predict, true_label)))

    # SVM预测
    for i in range(len(cnn_predictions)):
        cnn_predictions[i] = encode_one_hot(cnn_predictions[i], 5)
    if len(svm_predictions) != 0:
        for i in range(len(svm_predictions)):
            svm_predictions[i] = encode_one_hot(svm_predictions[i], 5)
        cnn_predictions = np.asarray(cnn_predictions) * 0.6
        svm_predictions = np.asarray(svm_predictions) * 0.4
        predictions = list()
        for cnn_pred in cnn_predictions:
            predictions.append(cnn_pred)
        for svm_pred in svm_predictions:
            predictions.append(svm_pred)
        predictions = np.asarray(predictions)
    else:
        predictions = cnn_predictions
    predictions = np.sum(predictions, axis=0)
    # print(predictions)
    predictions = decode_one_hot(predictions)
    # print(predictions)

    # print('average acc:{}'.format(get_acc(predictions, true_label)))
    predictions += 1
    # print(predictions)
    return predictions


def main():
    label = model_test('TFcodeX_test.tfrecord')


if __name__ == '__main__':
    main()
