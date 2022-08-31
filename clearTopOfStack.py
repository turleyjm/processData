import numpy as np
import os
import scyjava as sj
import scipy
from scipy import ndimage
import skimage as sm
import skimage.io
import tifffile

f = open("pythonText.txt", "r")

filename = "WoundXLJNK04"
stackFile = f"datProcessing/{filename}/{filename}.tif"
stack = sm.io.imread(stackFile).astype(int)
T, Z, C, X, Y = stack.shape
zero = np.zeros([T, Z, C, X, Y])

# Clear backwards
if True:
    # Settings Ecad Time:Depth

    time = [3, 6, 10, 12, 19, 23, 25, 29, 30, 40, 43, 44]
    depth = np.array([2, 3, 5, 6, 8, 10, 12, 14, 19, 22, 24, 26]) + 1

    for t, z in zip(time, depth):
        stack[t:, :z, 0] = zero[t:, :z, 0]

if True:
    # Settings H2 Time:Depth

    time = [3, 6, 10, 12, 19, 23, 25, 29, 30, 40, 43]
    depth = np.array([4, 5, 8, 9, 10, 14, 17, 23, 26, 28, 30]) + 1

    for t, z in zip(time, depth):
        stack[t:, :z, 1] = zero[t:, :z, 1]

# Clear forwards
if False:
    # Settings Ecad Time:Depth

    time = [0]
    depth = np.array([0]) + 1

    for t, z in zip(time, depth):
        stack[:t, :z, 0] = zero[:t, :z, 0]

if False:
    # Settings H2 Time:Depth

    time = [18, 31, 35, 92]
    depth = np.array([10, 7, 5, 4]) + 1

    for t, z in zip(time, depth):
        stack[:t, :z, 1] = zero[:t, :z, 1]


stack = np.asarray(stack, "uint8")
tifffile.imwrite(f"datProcessing/{filename}/clear{filename}.tif", stack, imagej=True)
