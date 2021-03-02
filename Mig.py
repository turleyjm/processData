import numpy as np
import os
import scyjava as sj
import scipy
from scipy import ndimage
import skimage as sm
import skimage.io
import tifffile
import utilBatch


f = open("pythonText.txt", "r")

filenames = f.read()
filenames = filenames.split(", ")


for filename in filenames:

    print(filename)

    stackFile = f"datProcessing/{filename}/{filename}.tif"
    stack = sm.io.imread(stackFile).astype(int)

    (T, Z, C, Y, X) = stack.shape

    for t in range(T):
        stack[t, :, 1] = ndimage.median_filter(stack[t, :, 1], size=(3, 3, 3))

    migration = np.asarray(stack[:, :, 1], "uint8")

    migration = utilBatch.normaliseMigration(migration, "MEDIAN", 10)
    tifffile.imwrite(f"datProcessing/{filename}/migration{filename}.tif", migration)