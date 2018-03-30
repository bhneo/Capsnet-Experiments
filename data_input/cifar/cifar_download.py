import os
import sys
from six.moves import urllib
import tarfile


# 从网址下载数据集存放到data_dir指定的目录下
CIFAR10_DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
CIFAR100_DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'


# 从网址下载数据集存放到data_dir指定的目录中
def maybe_download_and_extract(data_dir, data_url=CIFAR10_DATA_URL):
    """下载并解压缩数据集 from Alex's website."""
    dest_directory = data_dir  # '../CIFAR10_dataset'
    DATA_URL = data_url
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]  # 'cifar-10-binary.tar.gz'
    filepath = os.path.join(dest_directory, filename)  # '../CIFAR10_dataset\\cifar-10-binary.tar.gz'
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    # if data_url== CIFAR10_DATA_URL:
    #     extracted_dir_path = os.path.join(dest_directory,'cifar-10-batches-bin')  # '../CIFAR10_dataset\\cifar-10-batches-bin'
    # else :
    #     extracted_dir_path = os.path.join(dest_directory, 'cifar-100-binary')  # '../CIFAR10_dataset\\cifar-10-batches-bin'
    # if not os.path.exists(extracted_dir_path):
    #     tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    tarfile.open(filepath, 'r:gz').extractall(dest_directory)