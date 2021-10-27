import numpy as np
import os
import scyjava as sj
import scipy
from scipy import ndimage
import skimage as sm
import skimage.io
import tifffile

f = open("pythonText.txt", "r")

filename = f.read()
stackFile = f"datProcessing/{filename}/{filename}.tif"
stack = sm.io.imread(stackFile).astype(int)
T, Z, C, X, Y = stack.shape
zero = np.zeros([T, Z, C, X, Y])

if True:
    # Settings Ecad Time:Depth

    time = [27, 49, 56, 67, 73]
    depth = [1, 2, 3, 4, 5]  # (+1)

    for t, z in zip(time, depth):
        stack[t:, :z, 0] = zero[t:, :z, 0]

if True:
    # Settings H2 Time:Depth

    time = [27, 49, 56, 67, 73]
    depth = [1, 2, 3, 4, 5]  # (+1)

    for t, z in zip(time, depth):
        stack[t:, :z, 1] = zero[t:, :z, 1]

stack = np.asarray(stack, "uint8")
tifffile.imwrite(f"datProcessing/{filename}/clear{filename}.tif", stack)
