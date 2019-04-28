"""
Adapted by Charles Deledalle from https://gist.github.com/xmfbit
Inspired from http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""


def load(dataset="training", path=None):
    """
    Import either the training or testing MNIST data set.
    It returns a pair with the first element being the collection of
    images stacked in columns and the second element being a vector
    of corresponding labels from 0 to 9.

    Arguments:
        dataset (string, optional): either "training" or "testing".
            (default: "training")
        path (string, optional): the path pointing to the MNIST dataset
            If path=None, it looks succesively for the dataset at:
            '/datasets/MNIST' and './MNIST'. (default: None)

    Example:
        x, lbl = load(dataset="testing", path="/Folder/for/MNIST")
    """
    import os
    import struct
    import numpy as np

    if path is None:
        path = '/datasets/MNIST'
        if not os.path.isdir(path):
            path = './MNIST'
    if not os.path.isdir(path):
        raise ValueError("Cannot find dataset at '%s'" % path)

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows * cols)

    img = np.moveaxis(img, 0, -1)
    lbl = lbl.astype(int)

    return img, lbl


def show(image):
    """
    Render a given MNIST image provided as a column vector.

    Arguments:
        image (array): an array of shape (28*28) or (28, 28) representing a
            grey level image of size 28 x 28. Values are expected to be in the
            range [0, 1].

    Example:
        x, lbl = load(dataset="training", path="/datasets/MNIST")
        show(x[:, 0])
    """
    from matplotlib import pyplot
    import matplotlib as mpl

    rows = 28
    cols = 28
    if image.shape[0] != rows * cols and image.shape[0] * image.shape[1] != rows * cols:
        raise "the input is not an MNIST image."
    fig = pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    image = image.reshape(rows, cols)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()
