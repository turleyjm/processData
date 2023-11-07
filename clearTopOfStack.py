import numpy as np
import os
import functions
import scyjava as sj
import scipy
from scipy import ndimage
import skimage as sm
import skimage.io
import tifffile

filename = "WoundLrpr05"
stackFile = f"datProcessing/{filename}/{filename}.tif"
stack = sm.io.imread(stackFile).astype(int)
T, Z, C, X, Y = stack.shape
zero = np.zeros([T, Z, C, X, Y])
time = []
depth = []

# Clear backwards
if True:
    # Settings Ecad Time:Depth

    # time = [3, 32, 38, 50, 61]
    # depth = np.array([3, 5, 7, 9, 11]) + 1

    for t, z in zip(time, depth):
        stack[t:, :z, 0] = zero[t:, :z, 0]

if True:
    # Settings H2 Time:Depth

    # time = [26, 27]
    # depth = np.array([3, 9]) + 1

    for t, z in zip(time, depth):
        stack[t:, :z, 1] = zero[t:, :z, 1]

# Clear forwards
if False:
    # Settings Ecad Time:Depth

    time = [1, 3, 4, 5]
    depth = np.array([4, 3, 2, 1]) + 1

    for t, z in zip(time, depth):
        stack[:t, :z, 0] = zero[:t, :z, 0]

if False:
    # Settings H2 Time:Depth

    # time = [18, 31, 35, 92]
    # depth = np.array([10, 7, 5, 4]) + 1

    for t, z in zip(time, depth):
        stack[:t, :z, 1] = zero[:t, :z, 1]

focus = np.zeros([T, X, Y, 3])
focus[:, :, :, 1] = functions.focusStack(stack[:, :, 0], 7)[1]
focus[:, :, :, 0] = functions.focusStack(stack[:, :, 1], 7)[1]
focus[:, :, :, 1] = functions.normalise(focus[:, :, :, 1], "MEDIAN", 60)
focus[:, :, :, 0] = functions.normalise(focus[:, :, :, 0], "UPPER_Q", 90)

stack = np.asarray(stack, "uint8")
tifffile.imwrite(f"datProcessing/{filename}/clear{filename}.tif", stack, imagej=True)

focus = np.asarray(focus, "uint8")
tifffile.imwrite(
    f"datProcessing/{filename}/clearFocus{filename}.tif", focus, imagej=True
)
