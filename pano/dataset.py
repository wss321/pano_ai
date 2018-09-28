import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import random

file_dir = '../datas/pano'


def pares_tf(example_proto):
    features = {"id": tf.FixedLenFeature((), tf.int64),
                "data": tf.FixedLenFeature((256, 256), tf.float32),
                "label": tf.FixedLenFeature((), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["id"], tf.one_hot(parsed_features["label"] - 1, 5), parsed_features["data"]


def get_train_batch_iterator(batch_size, repeat=1, shuffle_buffer_size=10000):
    file_path = [os.path.join(file_dir, 'TFcodeX_{}.tfrecord'.format(i)) for i in range(1, 9)]
    pano = tf.data.TFRecordDataset([file_path])
    pano = pano.map(pares_tf)
    if shuffle_buffer_size:
        pano = pano.shuffle(buffer_size=shuffle_buffer_size)
    if repeat is None:
        pano = pano.repeat()
    else:
        pano = pano.repeat(repeat)
    pano = pano.batch(batch_size)
    iterator = pano.make_initializable_iterator()

    return iterator


def get_test_one_shot(shuffle_buffer_size=10000):
    file_path = [os.path.join(file_dir, 'TFcodeX_{}.tfrecord'.format(i)) for i in range(9, 11)]
    pano = tf.data.TFRecordDataset([file_path])
    pano = pano.map(pares_tf)
    if shuffle_buffer_size:
        pano = pano.shuffle(buffer_size=shuffle_buffer_size)

    return pano.make_one_shot_iterator().get_next()


def get_train_one_shot(shuffle_buffer_size=10000):
    file_path = [os.path.join(file_dir, 'TFcodeX_{}.tfrecord'.format(i)) for i in range(1, 9)]
    pano = tf.data.TFRecordDataset([file_path])
    pano = pano.map(pares_tf)
    if shuffle_buffer_size:
        pano = pano.shuffle(buffer_size=shuffle_buffer_size)

    return pano.make_one_shot_iterator().get_next()


def load_test_data_3C():
    test_iterator = get_test_one_shot()
    labels = []
    test_data = []
    with tf.Session() as sess:
        while True:
            try:
                id, one_hot_label, img = sess.run(test_iterator)
                labels.append(one_hot_label)
                img = np.asarray(img).reshape(256, 256, 1)
                test_data.append(img.repeat(3, axis=2))
            except:
                break
    test_data = np.asarray(test_data)
    return test_data, np.asarray(labels)


def load_train_data_3C():
    test_iterator = get_train_one_shot()
    labels = []
    train_data = []
    with tf.Session() as sess:
        while True:
            try:
                id, one_hot_label, img = sess.run(test_iterator)
                labels.append(one_hot_label)
                img = np.asarray(img).reshape(256, 256, 1)
                train_data.append(img.repeat(3, axis=2))
            except:
                break
    train_data = np.asarray(train_data)
    return train_data, np.asarray(labels)


def load_train_data_1C():
    test_iterator = get_train_one_shot()
    labels = []
    test_data = []
    with tf.Session() as sess:
        while True:
            try:
                id, one_hot_label, img = sess.run(test_iterator)
                labels.append(one_hot_label)
                test_data.append(img)
            except:
                break
    return np.asarray(test_data).reshape(-1, 256, 256, 1), np.asarray(labels)


def load_test_data_1C():
    test_iterator = get_test_one_shot()
    labels = []
    test_data = []
    with tf.Session() as sess:
        while True:
            try:
                id, one_hot_label, img = sess.run(test_iterator)
                labels.append(one_hot_label)
                test_data.append(img)
            except:
                break
    return np.asarray(test_data).reshape(-1, 256, 256, 1), np.asarray(labels)


def image_array_to_image_matrix(image_array):
    """Gets a image matrix from CIFAR-100 and turns it into a matrix suitable for CNN Networks"""
    image_size = int(np.sqrt(np.prod(np.shape(image_array)) / 1))

    image_matrix = image_array.reshape(image_size, image_size, 1)
    return image_matrix


def visualize_image(image, visual=True, title=''):
    """Displays the image in a zjl-230 matrix"""
    num_dims = len(np.shape(image))
    if num_dims == 1:
        image_to_visualize = image_array_to_image_matrix(image)
    elif num_dims == 3:
        image_to_visualize = image
    else:
        raise ValueError("Image array should have one or three dimensions")
    img = image_to_visualize.reshape(256, 256)
    if visual:
        plt.figure()
        plt.title(title)
        plt.imshow(img, interpolation='nearest')
        plt.show()
    return img


def _get_classify_data():
    """"""

    def get_train_one_shot(shuffle_buffer_size=10000):
        file_path = [os.path.join(file_dir, 'TFcodeX_{}.tfrecord'.format(i)) for i in range(1, 2)]
        pano = tf.data.TFRecordDataset([file_path])
        pano = pano.map(pares_tf)
        if shuffle_buffer_size:
            pano = pano.shuffle(buffer_size=shuffle_buffer_size)

        return pano.make_one_shot_iterator().get_next()

    train_iterator = get_train_one_shot()
    labels = []
    train_data = []
    with tf.Session() as sess:
        while True:
            try:
                id, one_hot_label, img = sess.run(train_iterator)
                labels.append(one_hot_label)
                train_data.append(np.asarray([img, img, img]).transpose((2, 0, 1)))
            except:
                break
    train_class_data = {'1': [], '2': [], '3': [], '4': [], '5': []}
    for index, value in enumerate(labels):
        for k, j in enumerate(value):
            if j == 1.0:
                label = k + 1
                train_class_data[str(label)].append(train_data[index])
                break
    return train_class_data


def get_sample_train_data(sample_num):
    """
    分离出 关系网络的sample集 和 train 集
    :param sample_num: sample 个数
    :return:
    """
    train_class_data = _get_classify_data()
    idx = random.randint(0, 10)
    sample_data = []
    sample_label = []
    train_data = []
    train_label = []
    for i in range(1, 6):
        for data in train_class_data[str(i)][idx:idx + sample_num]:
            sample_data.append(data)
        for data in train_class_data[str(i)][idx + sample_num:]:
            train_data.append(data)
        # print(np.asarray(train_data).shape)
        label = [0.0, 0.0, 0.0, 0.0, 0.0]
        label[i - 1] = 1.0
        for num in range(sample_num):
            sample_label.append(label)
        for num in range(len(train_class_data[str(i)][idx + sample_num:])):
            train_label.append(label)
    train_class_data = None
    sample_data = np.asarray(sample_data).reshape(sample_num * 5, 256, 256, 3)
    sample_label = np.asarray(
        sample_label).reshape(
        sample_num * 5, 5)
    train_data = np.asarray(train_data).reshape(-1, 256, 256, 3)
    train_label = np.asarray(train_label).reshape(-1, 5)
    return sample_data, sample_label, train_data, train_label


if __name__ == '__main__':
    # test_data, test_label = load_test_data_3C()
    # train_data, train_label = load_train_data_1C()
    # # print(test_data, test_label)
    # print(train_data, train_label)
    # train_label_sum = [0, 0, 0, 0, 0]
    # for i in train_label:
    #     for k, j in enumerate(i):
    #         if j == 1.0:
    #             train_label_sum[k] += 1
    #             break
    # print(train_label)
    # print(train_label_sum)
    # plt.figure()
    # for i in range(20):
    #     for idx, value in enumerate(train_label[i]):
    #         if value == 1.0:
    #             label = idx + 1
    #             break
    #     img = visualize_image(train_data[i], visual=False, title='index:{} - label:{}'.format(i, label))

    # train_class_data = get_classify_data()
    # for i in range(5):
    #     print(len(train_class_data[str(i + 1)]))
    sample, s_label, train, t_label = get_sample_train_data(20)

    print(
        sample.shape,
        s_label.shape,
        train.shape,
        t_label.shape,
    )
    print()
    print(s_label)
