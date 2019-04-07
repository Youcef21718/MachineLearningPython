import os
import functools
import operator
import gzip
import struct
import array
import tempfile
import numpy as np

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve  # py2
try:
    from urllib.parse import urljoin
except ImportError:
    from urlparse import urljoin

# the url can be changed by the users of the library (not a constant)
datasets_url = 'http://yann.lecun.com/exdb/mnist/'


class IdxDecodeError(ValueError):
    """Raised when an invalid idx file is parsed."""
    pass


def download_file(fname, target_dir=None, force=False):
    """Download fname from the datasets_url, and save it to target_dir,
    unless the file already exists, and force is False.
    Parameters
    ----------
    fname : str
        Name of the file to download
    target_dir : str
        Directory where to store the file
    force : bool
        Force downloading the file, if it already exists
    Returns
    -------
    fname : str
        Full path of the downloaded file
    """
    if not target_dir:
        target_dir = tempfile.gettempdir()
    target_fname = os.path.join(target_dir, fname)

    if force or not os.path.isfile(target_fname):
        url = urljoin(datasets_url, fname)
        urlretrieve(url, target_fname)

    return target_fname


def parse_idx(fd):
    """Parse an IDX file, and return it as a numpy array.
    Parameters
    ----------
    fd : file
        File descriptor of the IDX file to parse
    endian : str
        Byte order of the IDX file. See [1] for available options
    Returns
    -------
    data : numpy.ndarray
        Numpy array with the dimensions and the data in the IDX file
    1. https://docs.python.org/3/library/struct.html#byte-order-size-and-alignment
    """
    DATA_TYPES = {0x08: 'B',  # unsigned byte
                  0x09: 'b',  # signed byte
                  0x0b: 'h',  # short (2 bytes)
                  0x0c: 'i',  # int (4 bytes)
                  0x0d: 'f',  # float (4 bytes)
                  0x0e: 'd'}  # double (8 bytes)

    header = fd.read(4)
    if len(header) != 4:
        raise IdxDecodeError('Invalid IDX file, file empty or does not contain a full header.')

    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

    if zeros != 0:
        raise IdxDecodeError('Invalid IDX file, file must start with two zero bytes. '
                             'Found 0x%02x' % zeros)

    try:
        data_type = DATA_TYPES[data_type]
    except KeyError:
        raise IdxDecodeError('Unknown data type 0x%02x in IDX file' % data_type)

    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                    fd.read(4 * num_dimensions))

    data = array.array(data_type, fd.read())
    data.byteswap()  # looks like array.array reads data as little endian

    expected_items = functools.reduce(operator.mul, dimension_sizes)
    if len(data) != expected_items:
        raise IdxDecodeError('IDX file has wrong number of items. '
                             'Expected: %d. Found: %d' % (expected_items, len(data)))

    return np.array(data).reshape(dimension_sizes)


def download_and_parse_mnist_file(fname, target_dir=None, force=False):
    """Download the IDX file named fname from the URL specified in dataset_url
    and return it as a numpy array.
    Parameters
    ----------
    fname : str
        File name to download and parse
    target_dir : str
        Directory where to store the file
    force : bool
        Force downloading the file, if it already exists
    Returns
    -------
    data : numpy.ndarray
        Numpy array with the dimensions and the data in the IDX file
    """
    fname = download_file(fname, target_dir=target_dir, force=force)
    fopen = gzip.open if os.path.splitext(fname)[1] == '.gz' else open
    with fopen(fname, 'rb') as fd:
        return parse_idx(fd)


def train_images():
    """Return train images from Yann LeCun MNIST database as a numpy array.
    Download the file, if not already found in the temporary directory of
    the system.
    Returns
    -------
    train_images : numpy.ndarray
        Numpy array with the images in the train MNIST database. The first
        dimension indexes each sample, while the other two index rows and
        columns of the image
    """
    return download_and_parse_mnist_file('train-images-idx3-ubyte.gz')


def test_images():
    """Return test images from Yann LeCun MNIST database as a numpy array.
    Download the file, if not already found in the temporary directory of
    the system.
    Returns
    -------
    test_images : numpy.ndarray
        Numpy array with the images in the train MNIST database. The first
        dimension indexes each sample, while the other two index rows and
        columns of the image
    """
    return download_and_parse_mnist_file('t10k-images-idx3-ubyte.gz')


def train_labels():
    """Return train labels from Yann LeCun MNIST database as a numpy array.
    Download the file, if not already found in the temporary directory of
    the system.
    Returns
    -------
    train_labels : numpy.ndarray
        Numpy array with the labels 0 to 9 in the train MNIST database.
    """
    return download_and_parse_mnist_file('train-labels-idx1-ubyte.gz')


def test_labels():
    """Return test labels from Yann LeCun MNIST database as a numpy array.
    Download the file, if not already found in the temporary directory of
    the system.
    Returns
    -------
    test_labels : numpy.ndarray
        Numpy array with the labels 0 to 9 in the train MNIST database.
    """
    return download_and_parse_mnist_file('t10k-labels-idx1-ubyte.gz')

import matplotlib.pyplot as plt

def show(array, index):
    plt.imshow(array[index], cmap=plt.get_cmap('gray_r'))
    plt.show()

def sig(m):
    return (np.exp(m))/(1+np.exp(m))

def dsig(m):
    return sig(m)*(1-sig(m))

def accuracy(sL1, B3, B2, W3, W2, Y):
    L2 = sL1.dot(W2)+B2
    sL2 = sig(L2)

    L3 = sL2.dot(W3)+B3
    sL3 = sig(L3)

    labels = np.argmax(sL3, axis=1)

    return np.sum(labels == Y)

def cost(sL1, B3, B2, W3, W2, Y):
    L2 = sL1.dot(W2)+B2
    sL2 = sig(L2)

    L3 = sL2.dot(W3)+B3
    sL3 = sig(L3)

    return np.sum(0.5*(sL3-Y)*(sL3-Y))

# hyperparameters
batch = 10
numEpochs = 10;
epochTime = 60000/batch
learningRate = 3/batch

def train(Labels, Data, B3, B2, W3, W2):
    for i in range(numEpochs):
        print(accuracy(Data, B3, B2, W3, W2, train_labels()))
        for j in range(int(epochTime)):

            nums = np.arange(60000)
            np.random.shuffle(nums)
            shuffle = nums[:batch]

            Y = Labels[shuffle]
            sL1 = Data[shuffle]

            # neural network

            L2 = sL1.dot(W2)+B2
            sL2 = sig(L2)

            L3 = sL2.dot(W3)+B3
            sL3 = sig(L3)

            # gradients

            dB3 = (sL3-Y)*dsig(L3)
            dB2 = dB3.dot(W3.T)*dsig(L2)

            dW3 = (sL2.T).dot(dB3)
            dW2 = (sL1.T).dot(dB2)

            # update

            B3 -= learningRate*dB3.sum(axis=0)
            B2 -= learningRate*dB2.sum(axis=0)

            W3 -= learningRate*dW3
            W2 -= learningRate*dW2

W12 = np.random.normal(0, 1/np.sqrt(784), (784,150))
W25 = np.random.normal(0, 1/np.sqrt(150), (150,10))

W23 = np.random.normal(0, 1/np.sqrt(150), (150,100))
W35 = np.random.normal(0, 1/np.sqrt(100), (100,10))

W34 = np.random.normal(0, 1/np.sqrt(100), (100,50))
W45 = np.random.normal(0, 1/np.sqrt(50), (50,10))


B12 = np.random.normal(0, 1, (150))
B25 = np.random.normal(0, 1, (10))

B23 = np.random.normal(0, 1, (100))
B35 = np.random.normal(0, 1, (10))

B34 = np.random.normal(0, 1, (50))
B45 = np.random.normal(0, 1, (10))


Y = np.zeros((60000,10))
Y[np.arange(60000), train_labels()] = 1

sL1 = train_images().reshape(60000,784)/256
train(Y, sL1, B25, B12, W25, W12)
print(accuracy(sL1, B25, B12, W25, W12, train_labels()))

sL2 = sig(sL1.dot(W12) + B12)
train(Y, sL2, B35, B23, W35, W23)
print(accuracy(sL2, B35, B23, W35, W23, train_labels()))

sL3 = sig(sL2.dot(W23) + B23)
train(Y, sL3, B45, B34, W45, W34)
print(accuracy(sL3, B45, B34, W45, W34, train_labels()))

# label
testLabels = np.zeros((10000,10))
testLabels[np.arange(10000), test_labels()] = 1

# input
testImages = test_images().reshape(10000,784)/256

accuracy(testImages, B23, B12, W23, W12, test_labels())

plt.imshow(trainImages[0].reshape(28,28),cmap=plt.get_cmap('gray_r'))
