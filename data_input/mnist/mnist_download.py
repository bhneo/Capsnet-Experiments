import os
import sys
import gzip
import shutil
from six.moves import urllib

import tensorflow as tf


tf.app.flags.DEFINE_string('dir', 'data/mnist', 'directory')
tf.app.flags.DEFINE_string('dataset', 'mnist', 'mnist or fashion-mnist')
tf.app.flags.DEFINE_bool('force', False, 'rewrite or not')
FLAGS = tf.app.flags.FLAGS


# mnist dataset
HOMEPAGE = "http://yann.lecun.com/exdb/mnist/"
MNIST_TRAIN_IMGS_URL = HOMEPAGE + "train-images-idx3-ubyte.gz"
MNIST_TRAIN_LABELS_URL = HOMEPAGE + "train-labels-idx1-ubyte.gz"
MNIST_TEST_IMGS_URL = HOMEPAGE + "t10k-images-idx3-ubyte.gz"
MNIST_TEST_LABELS_URL = HOMEPAGE + "t10k-labels-idx1-ubyte.gz"

# fashion-mnist dataset
HOMEPAGE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FASHION_MNIST_TRAIN_IMGS_URL = HOMEPAGE + "train-images-idx3-ubyte.gz"
FASHION_MNIST_TRAIN_LABELS_URL = HOMEPAGE + "train-labels-idx1-ubyte.gz"
FASHION_MNIST_TEST_IMGS_URL = HOMEPAGE + "t10k-images-idx3-ubyte.gz"
FASHION_MNIST_TEST_LABELS_URL = HOMEPAGE + "t10k-labels-idx1-ubyte.gz"


def download_and_uncompress_zip(URL, dataset_dir, force=False):
    '''
    Args:
        URL: the download links for data
        dataset_dir: the path to save data
        force: redownload data
    '''
    filename = URL.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    extract_to = os.path.splitext(filepath)[0]

    def download_progress(count, block_size, total_size):
        sys.stdout.write("\r>> Downloading %s %.1f%%" % (filename, float(count * block_size) / float(total_size) * 100.))
        sys.stdout.flush()

    if not force and os.path.exists(filepath):
        print("file %s already exist" % (filename))
    else:
        filepath, _ = urllib.request.urlretrieve(URL, filepath, download_progress)
        print()
        print('Successfully Downloaded', filename)

    # with zipfile.ZipFile(filepath) as fd:
    with gzip.open(filepath, 'rb') as f_in, open(extract_to, 'wb') as f_out:
        print('Extracting ', filename)
        shutil.copyfileobj(f_in, f_out)
        print('Successfully extracted')
        print()


def start_download(dir, dataset, force):
    if dataset == 'mnist':
        print("Start downloading dataset MNIST:")
        download_and_uncompress_zip(MNIST_TRAIN_IMGS_URL, dir, force)
        download_and_uncompress_zip(MNIST_TRAIN_LABELS_URL, dir, force)
        download_and_uncompress_zip(MNIST_TEST_IMGS_URL, dir, force)
        download_and_uncompress_zip(MNIST_TEST_LABELS_URL, dir, force)
    elif dataset == 'fashion-mnist':
        print("Start downloading dataset Fashion MNIST:")
        download_and_uncompress_zip(FASHION_MNIST_TRAIN_IMGS_URL, dir, force)
        download_and_uncompress_zip(FASHION_MNIST_TRAIN_LABELS_URL, dir, force)
        download_and_uncompress_zip(FASHION_MNIST_TEST_IMGS_URL, dir, force)
        download_and_uncompress_zip(FASHION_MNIST_TEST_LABELS_URL, dir, force)
    else:
        raise Exception("Invalid dataset name! please check it: ", dataset)


def main(_):
    start_download(FLAGS.dir, FLAGS.dataset, FLAGS.force)


if __name__ == "__main__":
    tf.app.run()