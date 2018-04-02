import os
import sys
import gzip
import shutil
from six.moves import urllib
import tensorflow as tf


tf.app.flags.DEFINE_string('dir', 'data/mnist', 'directory')
tf.app.flags.DEFINE_bool('force', False, 'rewrite or not')
FLAGS = tf.app.flags.FLAGS


# smallNORB dataset
HOMEPAGE = "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/"
SMALLNORB_TRAIN_DAT_URL = HOMEPAGE + "smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz"
SMALLNORB_TRAIN_CAT_URL = HOMEPAGE + "smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz"
SMALLNORB_TRAIN_INFO_URL = HOMEPAGE + "smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz"
SMALLNORB_TEST_DAT_URL = HOMEPAGE + "smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz"
SMALLNORB_TEST_CAT_URL = HOMEPAGE + "smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz"
SMALLNORB_TEST_INFO_URL = HOMEPAGE + "smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz"


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


def start_download(dir, force):
    print("Start downloading dataset small NORB:")
    download_and_uncompress_zip(SMALLNORB_TRAIN_DAT_URL, dir, force)
    download_and_uncompress_zip(SMALLNORB_TRAIN_CAT_URL, dir, force)
    download_and_uncompress_zip(SMALLNORB_TRAIN_INFO_URL, dir, force)
    download_and_uncompress_zip(SMALLNORB_TEST_DAT_URL, dir, force)
    download_and_uncompress_zip(SMALLNORB_TEST_CAT_URL, dir, force)
    download_and_uncompress_zip(SMALLNORB_TEST_INFO_URL, dir, force)


def main(_):
    start_download(FLAGS.dir, FLAGS.force)


if __name__ == "__main__":
    tf.app.run()
