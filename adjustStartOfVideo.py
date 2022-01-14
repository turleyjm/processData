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
T, Z, C, Y, X = stack.shape

# Negative
shift_x = -7
shift_y = 84
t = 0

if shift_x < 0:
    if shift_y < 0:
        stack[t, :, :, 0 : Y + shift_y, 0 : X + shift_x] = stack[
            t, :, :, -shift_y:Y, -shift_x:X
        ]
        stack[t, :, :, 0:Y, X + shift_x : X] = 0
        stack[t, :, :, Y + shift_y : Y, 0:X] = 0
    else:
        stack[t, :, :, shift_y:Y, 0 : X + shift_x] = stack[
            t, :, :, 0 : Y - shift_y, -shift_x:X
        ]
        stack[t, :, :, 0:Y, X + shift_x : X] = 0
        stack[t, :, :, 0:shift_y, 0:X] = 0
else:
    if shift_y < 0:
        stack[t, :, :, 0 : Y + shift_y, shift_x:X] = stack[
            t, :, :, -shift_y:Y, 0 : X - shift_x
        ]
        stack[t, :, :, 0:Y, 0:shift_x] = 0
        stack[t, :, :, Y + shift_y : Y, 0:X] = 0
    else:
        stack[t, :, :, shift_y:Y, shift_x:X] = stack[
            t, :, :, 0 : Y - shift_y, 0 : X - shift_x
        ]
        stack[t, :, :, 0:Y, 0:shift_x] = 0
        stack[t, :, :, 0:shift_y, 0:X] = 0

stack = np.asarray(stack, "uint8")
tifffile.imwrite(f"datProcessing/{filename}/shift{filename}.tif", stack, imagej=True)
