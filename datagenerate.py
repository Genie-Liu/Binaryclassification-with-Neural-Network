from PIL import Image
import numpy as np
import os
import os.path


def generateSimpleImage(path):
    """
    transform the images to black and white Image

    """
    files = os.listdir(path)

    i = 0
    for f in files:
        if not os.path.isdir(f):
            nim = trans(os.path.join(path, f))
            nim.save(os.path.join(path, str(i)+".jpg"))
            i += 1


def trans(filename,  width=30, height=30):
    """
    Transform normal image to simple image

    filename: image filename
    (width, heigth): the size of wanted image. Default size:(30, 30)
    """

    im = Image.open(filename)
    # Transform to black and white image with wanted size
    nim = im.convert("1").resize((width, height))
    return nim


def generateNNInputs(path):
    """
    Generate input data for Neural Network

    path: the source file path
    """

    files = os.listdir(path)
    X = []
    for f in files:
        if not os.path.isdir(f):
            im = Image.open(os.path.join(path, f))
            xx = list(im.getdata())
            xx = np.array(xx).reshape(1, -1).squeeze()
            X.append(xx)

    return np.array(X).T/255
