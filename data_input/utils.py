import os
import sys
import scipy
import numpy as np
import tensorflow as tf


def create_train_set(dataset, batch_size=128, n_repeat=-1):
    tr_image, tr_label, val_image, val_label, num_label, num_batch = load_data(dataset, batch_size, is_training=True)

    tr_data_set = tf.data.Dataset.from_tensor_slices((tr_image, tr_label)).repeat(n_repeat).batch(batch_size)
    val_data_set = tf.data.Dataset.from_tensor_slices((val_image, val_label)).repeat(n_repeat).batch(batch_size)

    handle = tf.placeholder(tf.string, [])
    feed_iterator = tf.data.Iterator.from_string_handle(handle, tr_data_set.output_types,
                                                        tr_data_set.output_shapes)
    images, labels = feed_iterator.get_next()
    # 创建不同的iterator
    train_iterator = tr_data_set.make_one_shot_iterator()
    val_iterator = val_data_set.make_initializable_iterator()

    return images, labels, train_iterator, val_iterator, handle, num_label, num_batch


def create_test_set(dataset, batch_size=128):
    te_image, te_label, num_label, num_batch = load_data(dataset, is_training=False)
    te_data_set = tf.data.Dataset.from_tensor_slices((te_image, te_label)).batch(batch_size)
    return te_data_set.make_one_shot_iterator().get_next(), num_label, num_batch


def extract_cifar(files):
    # 验证文件是否存在
    for f in files:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    # 读取数据
    labels = None
    images = None

    for f in files:
        bytestream = open(f, 'rb')
        # 读取数据
        buf = bytestream.read(10000 * (32 * 32 * 3 + 1))
        # 把数据流转化为np的数组
        data = np.frombuffer(buf, dtype=np.uint8)
        # 改变数据格式
        data = data.reshape(10000, 1 + 32 * 32 * 3)
        # 分割数组
        labels_images = np.hsplit(data, [1])

        label = labels_images[0].reshape(10000)
        image = labels_images[1].reshape(10000, 32, 32, 3)

        if labels is None:
            labels = label
            images = image
        else:
            # 合并数组，不能用加法
            labels = np.concatenate((labels, label))
            images = np.concatenate((images, image))

    return labels, images


def load_cifar10(batch_size, is_training=True, valid_size=0.1):
    if is_training:
        files = [os.path.join(sys.path[0], 'data_input', 'data/cifar10/cifar-10-batches-bin', 'data_batch_%d.bin' % i) for i in range(1, 6)]
        labels, images = extract_cifar(files)

        indices = np.random.permutation(50000)
        valid_idx, train_idx = indices[:50000 * valid_size], indices[50000 * valid_size:]

        batch_num = 50000 * (1 - valid_size) // batch_size

        return images[train_idx], labels[train_idx], images[valid_idx], labels[valid_idx], 10, batch_num
    else:
        files = [os.path.join('data/cifar10/cifar-10-batches-bin', 'test_batch.bin'), ]
        return extract_cifar(files), 10, 10000 // batch_size


def load_mnist(batch_size, is_training=True):
    path = os.path.join(sys.path[0], 'data_input', 'data', 'mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        images = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        labels = loaded[8:].reshape(60000).astype(np.int32)

        tr_x = images[:55000] / 255.
        tr_y = labels[:55000]

        val_x = images[55000:, ] / 255.
        val_y = labels[55000:]

        batch_num = 55000 // batch_size

        return tr_x, tr_y, val_x, val_y, 10, batch_num
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        te_x = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        te_y = loaded[8:].reshape(10000).astype(np.int32)

        return te_x / 255., te_y, 10, 10000 // batch_size


def load_fashion_mnist(batch_size, is_training=True):
    path = os.path.join(sys.path[0], 'data_input', 'data', 'fashion-mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        tr_images = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        tr_labels = loaded[8:].reshape(60000).astype(np.int32)

        tr_x = tr_images[:55000] / 255.
        tr_y = tr_labels[:55000]

        val_x = tr_images[55000:, ] / 255.
        val_y = tr_labels[55000:]

        batch_num = 55000 // batch_size

        return tr_x, tr_y, val_x, val_y, 10, batch_num
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        te_x = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        te_y = loaded[8:].reshape(10000).astype(np.int32)

        return te_x / 255., te_y, 10, 10000 // batch_size


def load_smallNORB(batch_size, is_training=True):
    pass


def load_data(dataset, batch_size, is_training=True):
    if dataset == 'mnist':
        return load_mnist(batch_size, is_training)
    elif dataset == 'fashion-mnist':
        return load_fashion_mnist(batch_size, is_training)
    elif dataset == 'smallNORB':
        return load_smallNORB(batch_size, is_training)
    elif dataset == 'cifar10':
        return load_cifar10(batch_size, is_training)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)


def get_batch_data(dataset, batch_size, num_threads):
    if dataset == 'mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_mnist(batch_size, is_training=True)
    elif dataset == 'fashion-mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_fashion_mnist(batch_size, is_training=True)
    elif dataset == 'smallNORB':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_smallNORB(batch_size, is_training=True)
    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)

    return (X, Y)


def save_images(imgs, size, path):
    """
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    """
    imgs = (imgs + 1.) / 2  # inverse_transform
    return scipy.misc.imsave(path, mergeImgs(imgs, size))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs


def get_transformation_matrix_shape(in_pose_shape, out_pose_shape):
    return [out_pose_shape[0], in_pose_shape[0]]
