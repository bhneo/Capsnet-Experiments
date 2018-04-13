# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np


# Global constants describing the CIFAR-10 data set.

# 用于描述CiFar数据集的全局常量
IMAGE_SIZE = 32
IMAGE_DEPTH = 3
NUM_CLASSES_CIFAR10 = 10
NUM_CLASSES_CIFAR20 = 20
NUM_CLASSES_CIFAR100 = 100
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar10(filename_queue, coarse_or_fine=None):
    """Reads and parses examples from CIFAR10 data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.

    # cifar10 binary中的样本记录：3072=32x32x3
    # <1 x label><3072 x pixel>
    # ...
    # <1 x label><3072 x pixel>

    # 类型标签字节数
    label_bytes = 1  # 2 for CIFAR-100
    height = 32
    width = 32
    depth = 3

    # 图像字节数
    image_bytes = height * width * depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    # 每一条样本记录由 标签 + 图像 组成，其字节数是固定的。
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    # 创建一个固定长度记录读取器，读取一个样本记录的所有字节（label_bytes + image_bytes)
    # 由于cifar10中的记录没有header_bytes 和 footer_bytes,所以设置为0
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes, header_bytes=0, footer_bytes=0)

    # 调用读取器对象的read 方法返回一条记录
    _, byte_data = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    # 将一个字节组成的string类型的记录转换为长度为record_bytes，类型为unit8的一个数字向量
    uint_data = tf.decode_raw(byte_data, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    # 将一个字节代表了标签，我们把它从unit8转换为int32.
    label = tf.cast(tf.strided_slice(uint_data, [0], [label_bytes]), tf.int32)
    label.set_shape([1])

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    # 剩余的所有字节都是图像数据，把他从unit8转换为int32
    # 转为三维张量[depth，height，width]
    depth_major = tf.reshape(
        tf.strided_slice(uint_data, [label_bytes], [record_bytes]),
        [depth, height, width])
    # Convert from [depth, height, width] to [height, width, depth].
    # 把图像的空间位置和深度位置顺序由[depth, height, width] 转换成[height, width, depth]
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    return image, label


def read_cifar100(filename_queue, coarse_or_fine='fine'):
    """Reads and parses examples from CIFAR100 data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """
    height = 32
    width = 32
    depth = 3

    # cifar100中每个样本记录都有两个类别标签，每一个字节是粗略分类标签，
    # 第二个字节是精细分类标签：<1 x coarse label><1 x fine label><3072 x pixel>
    coarse_label_bytes = 1
    fine_label_bytes = 1

    # 图像字节数
    image_bytes = height * width * depth

    # 每一条样本记录由 标签 + 图像 组成，其字节数是固定的。
    record_bytes = coarse_label_bytes + fine_label_bytes + image_bytes

    # 创建一个固定长度记录读取器，读取一个样本记录的所有字节（label_bytes + image_bytes)
    # 由于cifar100中的记录没有header_bytes 和 footer_bytes,所以设置为0
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes, header_bytes=0, footer_bytes=0)

    # 调用读取器对象的read 方法返回一条记录
    _, byte_data = reader.read(filename_queue)

    # 将一系列字节组成的string类型的记录转换为长度为record_bytes，类型为unit8的一个数字向量
    uint_data = tf.decode_raw(byte_data, tf.uint8)

    # 将一个字节代表了粗分类标签，我们把它从unit8转换为int32.
    coarse_label = tf.cast(tf.strided_slice(record_bytes, [0], [coarse_label_bytes]), tf.int32)

    # 将二个字节代表了细分类标签，我们把它从unit8转换为int32.
    fine_label = tf.cast(tf.strided_slice(record_bytes, [coarse_label_bytes], [coarse_label_bytes + fine_label_bytes]),
                         tf.int32)

    if coarse_or_fine == 'fine':
        label = fine_label  # 100个精细分类标签
    else:
        label = coarse_label  # 100个粗略分类标签
    label.set_shape([1])

    # 剩余的所有字节都是图像数据，把他从一维张量[depth * height * width]
    # 转为三维张量[depth，height，width]
    depth_major = tf.reshape(
        tf.strided_slice(uint_data, [coarse_label_bytes + fine_label_bytes], [record_bytes]),
        [depth, height, width])

    # 把图像的空间位置和深度位置顺序由[depth, height, width] 转换成[height, width, depth]
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    return image, label


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, height, shuffle, channels_last=True):
    """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    if not channels_last:
        image = tf.transpose(image, [2, 0, 1])
    features = {
        'images': image,
        'labels': tf.one_hot(label, 10),
        'recons_image': image,
        'recons_label': label,
    }

    if shuffle:
        batched_features = tf.train.shuffle_batch(
            features,
            batch_size=batch_size,
            num_threads=16,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        batched_features = tf.train.batch(
            features,
            batch_size=batch_size,
            num_threads=1,
            capacity=min_queue_examples + 3 * batch_size)

    batched_features['labels'] = tf.reshape(batched_features['labels'],
                                            [batch_size, 10])
    batched_features['recons_label'] = tf.reshape(
        batched_features['recons_label'], [batch_size])
    batched_features['height'] = height
    batched_features['depth'] = 3
    batched_features['num_targets'] = 1
    batched_features['num_classes'] = 10

    # Display the training images in the visualizer.
    tf.summary.image('images', batched_features['images'])

    return batched_features


def _distort_resize(image, height, width):
    """Distorts input images for CIFAR training.

  Adds standard distortions such as flipping, cropping and changing brightness
  and contrast.

  Args:
    image: A float32 tensor with last dimmension equal to 3.
    image_size: The output image size after cropping.

  Returns:
    distorted_image: A float32 tensor with shape [image_size, image_size, 3].
  """
    distorted_image = tf.random_crop(image, [height, width, 3])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(
        distorted_image, lower=0.2, upper=1.8)
    distorted_image.set_shape([height, width, 3])
    return distorted_image


def inputs(cifar10or20or100, eval_data, data_dir, batch_size, distort=False):
    """使用Reader ops 读取数据集，用于CIFAR的评估

  输入参数:
    cifar10or20or100:指定要读取的数据集是cifar10 还是细分类的cifar100 ，或者粗分类的cifar100
    eval_data: True or False ,指示要读取的是训练集还是测试集
    data_dir: 指向CIFAR-10 或者 CIFAR-100 数据集的目录
    batch_size: 每个批次的图像数量
    distort:数据增强

  返回:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

    # 判断是读取cifar10 还是 cifar100（cifar100可分为20类或100类）
    if cifar10or20or100 == 10:
        read_cifar = read_cifar10
        coarse_or_fine = None
        if not eval_data:
            filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        else:
            filenames = [os.path.join(data_dir, 'test_batch.bin')]
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    if cifar10or20or100 == 20 or cifar10or20or100 == 100:
        read_cifar = read_cifar100
        if not eval_data:
            filenames = [os.path.join(data_dir, 'train.bin')]
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        else:
            filenames = [os.path.join(data_dir, 'test.bin')]
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    if cifar10or20or100 == 100:
        coarse_or_fine = 'fine'
    if cifar10or20or100 == 20:
        coarse_or_fine = 'coarse'

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # 根据文件名列表创建一个文件名队列
    filename_queue = tf.train.string_input_producer(filenames)

    # 从文件名队列的文件中读取样本
    float_image, label = read_cifar(filename_queue, coarse_or_fine=coarse_or_fine)

    # 要生成的目标图像的大小，在这里与原图像的尺寸保持一致
    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # 用于评估过程的图像数据预处理
    if distort:
        # 为图像添加padding = 4，图像尺寸变为[32+4,32+4],为后面的随机裁切留出位置
        padded_image = tf.image.resize_image_with_crop_or_pad(float_image, width + 4, height + 4)
        # 下面的这些操作为原始图像添加了很多不同的distortions，扩增了原始训练数据集

        resized_image = _distort_resize(float_image, height, width)
    else:
        # Crop the central [height, width] of the image.（其实这里并未发生裁剪）
        resized_image = tf.image.resize_image_with_crop_or_pad(float_image, width, height)

    # 数据集标准化操作：减去均值 + 方差标准化
    image = tf.image.per_image_standardization(resized_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    # 通过构造样本队列(a queue of examples)产生一个批次的图像和标签
    return _generate_image_and_label_batch(image, label,
                                           min_queue_examples, batch_size, height,
                                           shuffle=False if eval_data else True)


### load data by python ###

LABEL_SIZE = 1
PIXEL_DEPTH = 255
NUM_CLASSES = 10

TRAIN_NUM = 10000
TRAIN_NUMS = 50000
TEST_NUM = 10000


def extract_data(filenames):
    # 验证文件是否存在
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    # 读取数据
    labels = None
    images = None

    for f in filenames:
        bytestream = open(f, 'rb')
        # 读取数据
        buf = bytestream.read(TRAIN_NUM * (IMAGE_SIZE * IMAGE_SIZE * IMAGE_DEPTH + LABEL_SIZE))
        # 把数据流转化为np的数组
        data = np.frombuffer(buf, dtype=np.uint8)
        # 改变数据格式
        data = data.reshape(TRAIN_NUM, LABEL_SIZE + IMAGE_SIZE * IMAGE_SIZE * IMAGE_DEPTH)
        # 分割数组
        labels_images = np.hsplit(data, [LABEL_SIZE])

        label = labels_images[0].reshape(TRAIN_NUM)
        image = labels_images[1].reshape(TRAIN_NUM, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH)

        if labels is None:
            labels = label
            images = image
        else:
            # 合并数组，不能用加法
            labels = np.concatenate((labels, label))
            images = np.concatenate((images, image))

    # images = (images - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH  此处无需归一化，此操作将放在计算图中

    return labels, images


def extract_train_data(files_dir, valid_size=0.1):
    # 获得训练数据
    filenames = [os.path.join(files_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
    labels, images = extract_data(filenames)

    indices = np.random.permutation(TRAIN_NUMS)
    valid_idx, train_idx = indices[:TRAIN_NUMS * valid_size], indices[TRAIN_NUMS * valid_size:]

    return images[train_idx], images[valid_idx], labels[train_idx], labels[valid_idx]


def extract_test_data(files_dir):
    # 获得测试数据
    filenames = [os.path.join(files_dir, 'test_batch.bin'), ]
    return extract_data(filenames)


# 把稠密数据label[1,5...]变为[[0,1,0,0...],[...]...]
def dense_to_one_hot(labels_dense, num_classes):
    # 数据数量
    num_labels = labels_dense.shape[0]
    # 生成[0,1,2...]*10,[0,10,20...]
    index_offset = np.arange(num_labels) * num_classes
    # 初始化np的二维数组
    labels_one_hot = np.zeros((num_labels, num_classes))
    # 相对应位置赋值变为[[0,1,0,0...],[...]...]
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


class Cifar10DataSet(object):
    """docstring for Cifar10DataSet"""

    def __init__(self, data_dir):
        super(Cifar10DataSet, self).__init__()
        self.train_images, self.valid_images, self.train_labels, self.valid_labels = extract_train_data(
            os.path.join(data_dir, 'cifar10/cifar-10-batches-bin'))
        self.test_labels, self.test_images = extract_test_data(os.path.join(data_dir, 'cifar10/cifar-10-batches-bin'))

        print(self.train_labels.size)

        self.train_labels = dense_to_one_hot(self.train_labels, NUM_CLASSES)
        self.test_labels = dense_to_one_hot(self.test_labels, NUM_CLASSES)

        # epoch完成次数
        self.epochs_completed = 0
        # 当前批次在epoch中进行的进度
        self.index_in_epoch = 0

    def next_train_batch(self, batch_size):
        # 起始位置
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        # print "self.index_in_epoch: ",self.index_in_epoch
        # 完成了一次epoch
        if self.index_in_epoch > TRAIN_NUMS:
            # epoch完成次数加1
            self.epochs_completed += 1
            # print "self.epochs_completed: ",self.epochs_completed
            # 打乱数据顺序，随机性
            perm = np.arange(TRAIN_NUMS)
            np.random.shuffle(perm)
            self.train_images = self.train_images[perm]
            self.train_labels = self.train_labels[perm]
            start = 0
            self.index_in_epoch = batch_size
            # 条件不成立会报错
            assert batch_size <= TRAIN_NUMS

        end = self.index_in_epoch
        # print "start,end: ",start,end

        return self.train_images[start:end], self.train_labels[start:end]

    def valid_data(self):
        return self.valid_images, self.valid_labels

    def test_data(self):
        return self.test_images, self.test_labels
