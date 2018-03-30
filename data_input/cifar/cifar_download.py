import os
import sys
import tarfile
import tensorflow as tf

from six.moves import urllib

tf.app.flags.DEFINE_string('dir', '../../data/cifar10', 'directory')
tf.app.flags.DEFINE_integer('cifar', 10, 'cifar 10 or 100')
tf.app.flags.DEFINE_bool('b', True, 'is binary')
FLAGS = tf.app.flags.FLAGS


# 从网址下载数据集存放到data_dir指定的目录下
CIFAR10_BINARY_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
CIFAR100_BINARY_URL = 'http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'
CIFAR10_PYTHON_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR100_PYTHON_URL = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'


# 从网址下载数据集存放到data_dir指定的目录中
def maybe_download_and_extract(data_dir, cifar10or100=10, binary=True):
    if cifar10or100 == 10:
        URL = CIFAR10_BINARY_URL if binary else CIFAR10_PYTHON_URL
    elif cifar10or100 == 100:
        URL = CIFAR100_BINARY_URL if binary else CIFAR100_PYTHON_URL
    """下载并解压缩数据集 from Alex's website."""
    dest_directory = data_dir  # '../CIFAR10_dataset'
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = URL.split('/')[-1]  # 'cifar-10-binary.tar.gz'
    filepath = os.path.join(dest_directory, filename)  # '../CIFAR10_dataset\\cifar-10-binary.tar.gz'
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
    maybe_download_and_extract(FLAGS.dir, FLAGS.cifar, FLAGS.b)

if __name__ == "__main__":
    tf.app.run()