from ctypes import util
import os
import shutil
from math import dist, floor, log10

from collections import Counter
import matplotlib
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import seed
from random import random
from random import shuffle
from pandas.core import frame
import scipy as sp
import scipy.linalg as linalg
import scipy.ndimage as nd
import skimage as sm
import skimage.io
import skimage.measure
import tifffile
from skimage.draw import circle_perimeter
from scipy import optimize
import xml.etree.ElementTree as et

import commonLiberty as cl
import utils as util

plt.rcParams.update({"font.size": 16})


# -------------------

# filenames, fileType = cl.getFilesType()

# Weka
if False:
    seed(100)
    training = np.zeros([2 * len(filenames), 512, 512])
    i = 0
    for filename in filenames:

        vid = sm.io.imread(f"datProcessing/{filename}/focus{filename}.tif").astype(int)

        t = int(90 * random())
        training[2 * i : 2 * i + 2] = vid[t : t + 2, :, :, 1]
        i += 1

    training = np.asarray(training, "uint8")
    tifffile.imwrite(f"wekaBourdary.tif", training)

    filenames = [
        "Unwound18h13",
        "Unwound18h14",
        "Unwound18h15",
        "Unwound18h16",
        "Unwound18h17",
        "Unwound18h18",
        "WoundL18h10",
        "WoundL18h11",
        "WoundL18h13",
        "WoundL18h14",
        "WoundL18h15",
        "WoundS18h14",
        "WoundS18h15",
        "WoundS18h16",
        "WoundS18h17",
        "WoundS18h18",
        "WoundXL18h02",
        "WoundXL18h03",
        "WoundXL18h04",
        "WoundXL18h05",
    ]

    training = np.zeros([3 * len(filenames), 512, 512])
    i = 0
    for filename in filenames:

        vid = sm.io.imread(f"datProcessing/{filename}/focus{filename}.tif").astype(int)

        if i < 4:
            t = 60 + int(30 * random())
        else:
            t = int(30 * random())
        training[3 * i : 3 * i + 3] = vid[t : t + 3, :, :, 1]
        i += 1

    training = np.asarray(training, "uint8")
    tifffile.imwrite(f"wekaWoundBourdary.tif", training)

# deep learning ecad
if True:
    filenames, fileType = util.getFilesType()
    seed(100)
    filenames.remove("ecadDeep.tif")

    seg = np.zeros([3, 12 * len(filenames), 512, 512])
    zeros = np.zeros([512, 512])
    shuffle(filenames)
    i = 0
    for filename in filenames:
        vid = sm.io.imread(f"dat/{filename}/focus{filename}.tif").astype(int)
        binary = sm.io.imread(f"dat/{filename}/binary{filename}.tif").astype(int)
        outPlane = sm.io.imread(f"dat/{filename}/outPlane{filename}.tif").astype(int)

        t = int(15 * random()) + 1
        seg[1, i] = vid[t - 1, :, :, 1]
        seg[0, i] = zeros
        seg[2, i] = zeros
        i += 1
        seg[1, i] = vid[t, :, :, 1]
        seg[0, i] = binary[t]
        seg[2, i] = outPlane[t]
        i += 1
        seg[1, i] = vid[t + 1, :, :, 1]
        seg[0, i] = zeros
        seg[2, i] = zeros
        i += 1

        t = int(30 * random()) + 1
        seg[1, i] = vid[t - 1, :, :, 1]
        seg[0, i] = zeros
        seg[2, i] = zeros
        i += 1
        seg[1, i] = vid[t, :, :, 1]
        seg[0, i] = binary[t]
        seg[2, i] = outPlane[t]
        i += 1
        seg[1, i] = vid[t + 1, :, :, 1]
        seg[0, i] = zeros
        seg[2, i] = zeros
        i += 1

        t = int(88 * random()) + 1
        seg[1, i] = vid[t - 1, :, :, 1]
        seg[0, i] = zeros
        seg[2, i] = zeros
        i += 1
        seg[1, i] = vid[t, :, :, 1]
        seg[0, i] = binary[t]
        seg[2, i] = outPlane[t]
        i += 1
        seg[1, i] = vid[t + 1, :, :, 1]
        seg[0, i] = zeros
        seg[2, i] = zeros
        i += 1

        t = 70 + int(15 * random())
        seg[1, i] = vid[t - 1, :, :, 1]
        seg[0, i] = zeros
        seg[2, i] = zeros
        i += 1
        seg[1, i] = vid[t, :, :, 1]
        seg[0, i] = binary[t]
        seg[2, i] = outPlane[t]
        i += 1
        seg[1, i] = vid[t + 1, :, :, 1]
        seg[0, i] = zeros
        seg[2, i] = zeros
        i += 1

    seg = np.asarray(seg, "uint8")
    tifffile.imwrite(f"dat/ecadDeep.tif", seg, imagej=True)