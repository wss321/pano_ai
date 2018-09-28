import tensorflow as tf
from tensorflow.contrib.image import rotate
import numpy as np
import keras

NUM_CHANNELS = 1


def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)  # 亮度
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  # 饱和度
        image = tf.image.random_hue(image, max_delta=0.2)  # 色相
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # 对比度
    if color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    if color_ordering == 2:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    if color_ordering == 3:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    return tf.clip_by_value(image, 0.0, 1.0)


def random_rotate_image(image):
    """Do random rotate to a image (shape = (height,weigh,channels))"""
    with tf.device('/cpu:0'):
        angle = tf.random_uniform(shape=(1,), minval=-25, maxval=25)
        return rotate(image, angle)


def distort_image(image, image_size, resize):
    """Does random distortion at the training images to avoid overfitting"""
    # image = tf.image.resize_images(image, (resize, resize))
    # image = tf.random_crop(image, [image_size, image_size, 1])
    image = tf.image.random_flip_left_right(image)
    image = random_rotate_image(image)
    image = tf.image.random_brightness(image,
                                       max_delta=30)
    image = tf.image.random_contrast(image,
                                     lower=0.2, upper=1.8)
    image = distort_color(image, np.random.randint(4))
    # 随机边框裁剪
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    # 随机噪声
    image = tf.slice(image, bbox_begin, bbox_size)
    rand = np.random.randint(100)
    if rand < 20:
        noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=1.0, dtype=tf.float32)
        image = tf.add(image, noise)
    # float_image = tf.image.per_image_standardization(image)
    return image


def distorted_batch(batch, image_size, resize):
    """Creates a distorted image batch"""
    return tf.map_fn(lambda frame: distort_image(frame, image_size, resize), batch)


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, session, distort_op, x_pc, data_X, data_Y, batch_size=32, shape=(256, 256, 1),
                 n_classes=190, shuffle=True, distort=True):
        """
            Initialize.
        :param session: tensorflow sesion.
        :param distort_op: tensorflow operation for distorting images.
        :param x_pc: tensorflow placeholder for feed data.
        :param data_X: X
        :param data_Y: label
        :param batch_size: batch size
        :param shape: image shape
        :param n_classes: the number of classes in DataSet
        :param shuffle: shuffle the order
        :param distort: whether distort
        """
        self.shape = shape
        self.batch_size = batch_size
        self.data_X = data_X
        self.data_Y = data_Y
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.distort = distort
        self.session = session
        self.distort_op = distort_op
        self.x_pc = x_pc
        self.indexes = np.arange(len(self.data_X))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.data_X) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch_data_X = [self.data_X[k] for k in batch_indexs]
        batch_data_Y = [self.data_Y[k] for k in batch_indexs]

        # Generate data
        X, y = self.__data_generation(batch_data_X, batch_data_Y)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, X, Y):
        """Generates data containing batch_size samples"""
        # Generate data
        Xs = [b_X for b_X in X]
        if self.distort:
            Xs = self.session.run(self.distort_op, feed_dict={self.x_pc: Xs})
        Ys = [b_Y for b_Y in Y]
        return np.asarray(Xs).reshape(-1, self.shape[0], self.shape[1], self.shape[2]), np.asarray(Ys)
