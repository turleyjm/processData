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

plt.rcParams.update({"font.size": 16})


# -------------------

filenames, fileType = cl.getFilesType()

seed(100)
training = np.zeros([3 * len(filenames), 512, 512])
i = 0
for filename in filenames:

    vid = sm.io.imread(f"datProcessing/{filename}/focus{filename}.tif").astype(int)

    t = int(90 * random())
    training[3 * i : 3 * i + 3] = vid[t : t + 3, :, :, 1]
    i += 1

training = np.zeros([3 * len(filenames), 512, 512])
i = 0
for filename in filenames:

    vid = sm.io.imread(f"datProcessing/{filename}/focus{filename}.tif").astype(int)

    t = int(30 * random())
    training[3 * i : 3 * i + 3] = vid[t : t + 3, :, :, 1]
    i += 1


training = np.asarray(training, "uint8")
tifffile.imwrite(f"wekaWoundBourdary.tif", training)
