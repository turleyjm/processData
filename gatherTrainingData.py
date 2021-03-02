import os
import shutil
from math import floor, log10

from collections import Counter
import numpy as np
import pandas as pd
import skimage as sm
import skimage.io
import tifffile
import random


# -------------------


def getFiles():
    f = open("pythonText.txt", "r")
    filenames = f.read()
    filenames = filenames.split(", ")
    return filenames


filenames = getFiles()

training = []

for filename in filenames:

    vidFile = f"datProcessing/{filename}/h2Focus{filename}.tif"
    vid = sm.io.imread(vidFile).astype(int)

    t0 = int(random.uniform(0, 1) * 40)  # for woundsite

    for i in range(3):
        training.append(vid[t0 + i])

    t0 = int(random.uniform(0, 1) * 176)

    for i in range(3):
        training.append(vid[t0 + i])

training = np.asarray(training, "uint8")
tifffile.imwrite(f"datProcessing/TrainingData/h2FocusTrainingData.tif", training)

training = []

for filename in filenames:

    vidFile = f"datProcessing/{filename}/ecadFocus{filename}.tif"
    vid = sm.io.imread(vidFile).astype(int)

    t0 = int(random.uniform(0, 1) * 50)  # for woundsite

    for i in range(3):
        training.append(vid[t0 + i])

    t0 = int(random.uniform(0, 1) * 176)

    for i in range(3):
        training.append(vid[t0 + i])

training = np.asarray(training, "uint8")
tifffile.imwrite(f"datProcessing/TrainingData/ecadFocusTrainingData.tif", training)

# training = np.asarray(training, "uint8")
# tifffile.imwrite(f"dat/TrainingData/woundTrainingData.tif", training)

# overlay = np.zeros([len(filenames) * 6, 512, 512, 3])
# deepEcad = []
# deepProb = []
# deepBinary = []
# t = 0
# for filename in filenames:

#     vidEcad = sm.io.imread(f"dat/{filename}/focusEcad{filename}.tif").astype(int)
#     vidProb = sm.io.imread(f"dat/{filename}/probBoundary{filename}.tif").astype(float)
#     vidBinary = sm.io.imread(f"dat/{filename}/binaryBoundary{filename}.tif").astype(int)

#     t0 = int(random.uniform(0, 1) * 30)
#     overlay[t : t + 3, :, :, 1] = vidProb[t0 : t0 + 3] * 150
#     overlay[t : t + 3, :, :, 0] = vidBinary[t0 : t0 + 3]
#     deepEcad.append(vidEcad[t0 + 1])
#     deepProb.append(vidProb[t0 + 1])
#     deepBinary.append(vidBinary[t0 + 1])

#     t += 3

#     t0 = int(random.uniform(0, 1) * 175)
#     overlay[t : t + 3, :, :, 1] = vidProb[t0 : t0 + 3] * 150
#     overlay[t : t + 3, :, :, 0] = vidBinary[t0 : t0 + 3]
#     deepEcad.append(vidEcad[t0 + 1])
#     deepProb.append(vidProb[t0 + 1])
#     deepBinary.append(vidBinary[t0 + 1])

#     t += 3


# deepEcad = np.asarray(deepEcad, "uint8")
# tifffile.imwrite(f"dat/trainingData/deepEcad.tif", deepEcad)
# deepProb = np.asarray(deepProb, "float")
# tifffile.imwrite(f"dat/trainingData/deepProb.tif", deepProb)
# deepBinary = np.asarray(deepBinary, "uint8")
# tifffile.imwrite(f"dat/trainingData/deepBinary.tif", deepBinary)
# overlay = np.asarray(overlay, "uint8")
# tifffile.imwrite(f"dat/trainingData/deepOverlayBinary.tif", overlay)
