import numpy as np
import os
from os.path import exists
from numpy.core.numeric import False_
from PIL import Image
import scyjava as sj
import scipy
from scipy import ndimage
import skimage as sm
import skimage.io
import tifffile
from datetime import datetime
import shutil
from shapely.geometry import Polygon
import utils as util
import cellProperties as cell
import pandas as pd
from collections import Counter
import scipy as sp
import xml.etree.ElementTree as et

### IMPORTS ###
import configparser
from pathlib import Path

scale = 123.26 / 512


def current_time():
    now = datetime.now()
    currentTime = now.strftime("%H:%M:%S")
    return currentTime


def sortGrid(dfVelocity, x, y):

    xMin = x[0]
    xMax = x[1]
    yMin = y[0]
    yMax = y[1]

    dfxmin = dfVelocity[dfVelocity["X"] > xMin]
    dfx = dfxmin[dfxmin["X"] < xMax]

    dfymin = dfx[dfx["Y"] > yMin]
    df = dfymin[dfymin["Y"] < yMax]

    return df


def process_stack(ij, filename):

    print("Finding Surface")
    # print(current_time())

    stackFile = f"datProcessing/{filename}/{filename}.tif"
    stack = sm.io.imread(stackFile).astype(int)

    (T, Z, C, Y, X) = stack.shape

    migration = np.zeros([T, Z, Y, X])

    migration = normaliseMigration(stack[:, :, 1], "UPPER_Q", 60)

    stack[:, :, 1] = migration

    stack = np.asarray(stack, "uint8")
    tifffile.imwrite(f"datProcessing/{filename}/_{filename}.tif", stack, imagej=True)

    surface = getSurface(stack[:, :, 0])
    if True:
        surface = np.asarray(surface, "uint8")
        tifffile.imwrite(f"datProcessing/{filename}/surface{filename}.tif", surface)

    print("Filtering Height")
    # print(current_time())

    ecad = heightFilter(stack[:, :, 0], surface, 10)
    h2 = heightFilter(stack[:, :, 1], surface, 15)

    if True:
        ecad = np.asarray(ecad, "uint8")
        tifffile.imwrite(
            f"datProcessing/{filename}/ecadHeight{filename}.tif",
            ecad,
            imagej=True,
            metadata={"axes": "TZYX"},
        )
    if False:
        h2 = np.asarray(h2, "uint8")
        tifffile.imwrite(f"datProcessing/{filename}/h2Height{filename}.tif", h2)

    print("Focussing the image stack")
    # print(current_time())

    ecadFocus = focusStack(ecad, 7)[1]
    h2Focus = focusStack(h2, 7)[1]

    if False:
        ecadFocus = np.asarray(ecadFocus, "uint8")
        tifffile.imwrite(
            f"datProcessing/{filename}/ecadBleach{filename}.tif", ecadFocus
        )
        h2Focus = np.asarray(h2Focus, "uint8")
        tifffile.imwrite(f"datProcessing/{filename}/h2Bleach{filename}.tif", h2Focus)

    print("Normalising images")
    # print(current_time())

    ecadNormalise = normalise(ecadFocus, "MEDIAN", 60)
    h2Normalise = normalise(h2Focus, "UPPER_Q", 90)

    if False:
        ecadNormalise = np.asarray(ecadNormalise, "uint8")
        tifffile.imwrite(
            f"datProcessing/{filename}/ecadFocus{filename}.tif", ecadNormalise
        )
        h2Normalise = np.asarray(h2Normalise, "uint8")
        tifffile.imwrite(f"datProcessing/{filename}/h2Focus{filename}.tif", h2Normalise)

    focus = np.zeros([T, 512, 512, 3])

    for t in range(T):
        focus[t, :, :, 0] = h2Normalise[t]
        focus[t, :, :, 1] = ecadNormalise[t]

    focus = np.asarray(focus, "uint8")
    tifffile.imwrite(f"datProcessing/{filename}/focus{filename}.tif", focus)

    print("Migration Stack")
    # print(current_time())

    macro = """
        open("%s");
        main_win = getTitle();
        run("Duplicate...", "duplicate channels=2");
        H2_win = getTitle();
        selectWindow(main_win);
        close();
        selectWindow(H2_win);
        run("Median 3D...", "x=3 y=3 z=3");
        saveAs("Tiff", "%s");
    """ % (
        f"/Users/jt15004/Documents/Coding/python/processData/datProcessing/{filename}/_{filename}.tif",
        f"/Users/jt15004/Documents/Coding/python/processData/datProcessing/{filename}/migration{filename}.tif",
    )

    ij.script().run("macro.ijm", macro, True).get()
    image_ij2 = ij.py.active_dataset()

    os.remove(f"datProcessing/{filename}/_{filename}.tif")


def weka(
    ij,
    filename,
    model_path,
    name,
):

    framesMax = 5
    weka = sj.jimport("trainableSegmentation.WekaSegmentation")()
    weka.loadClassifier(model_path)

    ecadFile = f"datProcessing/{filename}/focus{filename}.tif"
    ecad = sm.io.imread(ecadFile).astype(int)
    ecad = ecad[:, :, :, 1]

    (T, X, Y) = ecad.shape

    if T == 8:
        j = 1
        print(f" part {j} -----------------------------------------------------------")
        stackprob = np.zeros([T, X, Y])
        stack = ecad[0:4]
        stack_ij2 = ij.py.to_dataset(stack)
        stackprob_ij2 = apply_weka(ij, weka, stack_ij2)
        stackprob[0:4] = ij.py.from_java(stackprob_ij2).values

        j += 1
        print(f" part {j} -----------------------------------------------------------")

        stack = ecad[4:]
        stack_ij2 = ij.py.to_dataset(stack)
        stackprob_ij2 = apply_weka(ij, weka, stack_ij2)
        stackprob[4:] = ij.py.from_java(stackprob_ij2).values

    else:
        path_to_file = (
            f"datProcessing/{filename}/_{name}{filename}/_{name}{filename}.tif"
        )
        if exists(path_to_file):
            stackprob = sm.io.imread(path_to_file).astype(int)
            startPoint = int(
                np.sum(np.amax(np.amax(stackprob, axis=1), axis=1) > 0) / framesMax
            )
            split = int(T / framesMax - startPoint)
        else:
            util.createFolder(f"datProcessing/{filename}/_{name}{filename}")
            stackprob = np.zeros([T, X, Y])

            split = int(T / framesMax - 1)
            stack = ecad[0:framesMax]
            stack_ij2 = ij.py.to_dataset(stack)
            stackprob_ij2 = apply_weka(ij, weka, stack_ij2)
            stackprob[0:framesMax] = ij.py.from_java(stackprob_ij2).values

            stackprob = np.asarray(stackprob, "uint8")
            tifffile.imwrite(
                f"datProcessing/{filename}/_{name}{filename}/_{name}{filename}.tif",
                stackprob,
            )
            startPoint = 1

        j = startPoint
        print(f" part {j} -----------------------------------------------------------")
        j += 1

        for i in range(split):
            stack = ecad[
                framesMax * (i + startPoint) : framesMax + framesMax * (i + startPoint)
            ]
            stack_ij2 = ij.py.to_dataset(stack)
            stackprob_ij2 = apply_weka(ij, weka, stack_ij2)
            stackprob[
                framesMax * (i + startPoint) : framesMax + framesMax * (i + startPoint)
            ] = ij.py.from_java(stackprob_ij2).values

            stackprob = np.asarray(stackprob, "uint8")
            tifffile.imwrite(
                f"datProcessing/{filename}/_{name}{filename}/_{name}{filename}.tif",
                stackprob,
            )

            print(
                f" part {j} -----------------------------------------------------------"
            )
            j += 1

        stack = ecad[framesMax * (T // framesMax) :]
        stack_ij2 = ij.py.to_dataset(stack)
        stackprob_ij2 = apply_weka(ij, weka, stack_ij2)
        stackprob[framesMax * (T // framesMax) :] = ij.py.from_java(
            stackprob_ij2
        ).values

        print(f" part {j} -----------------------------------------------------------")
        j += 1
        shutil.rmtree(f"datProcessing/{filename}/_{name}{filename}")

    stackprob = 255 - np.asarray(stackprob, "uint8")
    tifffile.imwrite(f"datProcessing/{filename}/{name}{filename}.tif", stackprob)


# -----------------


def getSurface(ecad):

    ecad = ecad.astype("float")
    variance = ecad
    (T, Z, Y, X) = ecad.shape

    for t in range(T):
        for z in range(Z):
            win_mean = ndimage.uniform_filter(ecad[t, z], (40, 40))
            win_sqr_mean = ndimage.uniform_filter(ecad[t, z] ** 2, (40, 40))
            variance[t, z] = win_sqr_mean - win_mean ** 2

    win_sqr_mean = 0
    win_mean = 0

    surface = np.zeros([T, X, Y])

    mu0 = 400
    for t in range(T):
        mu = variance[t, :, 50:450, 50:450][variance[t, :, 50:450, 50:450] > 0]

        mu = np.quantile(mu, 0.5)

        ratio = mu0 / mu

        variance[t] = variance[t] * ratio
        variance[t][variance[t] > 65536] = 65536

    np.argmax(variance[0] > 1000, axis=0)

    surface = np.argmax(variance > 1000, axis=1)

    surface = ndimage.median_filter(surface, size=9)

    return surface


def heightFilter(channel, surface, height):

    (T, Z, Y, X) = channel.shape

    heightFilt = np.zeros([T, Y, X, Z])

    b = np.arange(Z)
    heightFilt += b

    surfaceBelow = np.repeat(surface[:, :, :, np.newaxis], Z, axis=3) + height

    # heightFilt = np.einsum("ijkl->iljk", heightFilt)

    heightFilt = heightFilt < surfaceBelow
    heightFilt = heightFilt.astype(float)

    heightFilt = heightFilt.astype(float)
    heightFilt = np.einsum("ijkl->iljk", heightFilt)
    for t in range(T):
        for z in range(Z):
            heightFilt[t, z] = ndimage.uniform_filter(heightFilt[t, z], (20, 20))

    # heightFilt = np.asarray(heightFilt * 254, "uint8")
    # tifffile.imwrite(f"heightFilt.tif", heightFilt)

    channel = channel * heightFilt

    return channel


# Returns the full macro code with the filepath and focus range inserted as
# hard-coded values.


def focusStack(image, focusRange):

    image = image.astype("uint16")
    (T, Z, Y, X) = image.shape
    variance = np.zeros([T, Z, Y, X])
    varianceMax = np.zeros([T, Y, X])
    surface = np.zeros([T, Y, X])
    focus = np.zeros([T, Y, X])

    for t in range(T):
        for z in range(Z):
            winMean = ndimage.uniform_filter(image[t, z], (focusRange, focusRange))
            winSqrMean = ndimage.uniform_filter(
                image[t, z] ** 2, (focusRange, focusRange)
            )
            variance[t, z] = winSqrMean - winMean ** 2

    varianceMax = np.max(variance, axis=1)

    for z in range(Z):
        surface[variance[:, z] == varianceMax] = z

    for z in range(Z):
        focus[surface == z] = image[:, z][surface == z]

    surface = surface.astype("uint8")
    focus = focus.astype("uint8")

    return surface, focus


def normalise(vid, calc, mu0):
    vid = vid.astype("float")
    (T, X, Y) = vid.shape

    for t in range(T):
        mu = vid[t, 50:450, 50:450][vid[t, 50:450, 50:450] > 0]

        if calc == "MEDIAN":
            mu = np.quantile(mu, 0.5)
        elif calc == "UPPER_Q":
            mu = np.quantile(mu, 0.75)

        ratio = mu0 / mu

        vid[t] = vid[t] * ratio
        vid[t][vid[t] > 255] = 255

    return vid.astype("uint8")


def normaliseMigration(vid, calc, mu0):
    vid = vid.astype("float")
    (T, Z, X, Y) = vid.shape

    for t in range(T):
        mu = vid[t, :, 50:450, 50:450][vid[t, :, 50:450, 50:450] > 0]

        if calc == "MEDIAN":
            mu = np.quantile(mu, 0.5)
        elif calc == "UPPER_Q":
            mu = np.quantile(mu, 0.75)

        ratio = mu0 / mu

        vid[t] = vid[t] * ratio
        vid[t][vid[t] > 255] = 255

    return vid.astype("uint8")


def apply_weka(ij, classifier, image_ij2):

    ijf = sj.jimport("net.imglib2.img.display.imagej.ImageJFunctions")()

    image_ij1 = ijf.wrap(image_ij2, sj.to_java("IJ1 image"))

    prob_ij1 = classifier.applyClassifier(image_ij1, 6, True)

    n_channels = classifier.getNumOfClasses()
    n_frames = image_ij1.getNChannels()
    prob_ij1.setDimensions(n_channels, 1, n_frames)

    prob_ij2 = ij.py.to_dataset(prob_ij1)
    prob_np = ij.py.from_java(prob_ij2).astype("float16").values * 255
    prob_np = prob_np.astype("uint8")[:, 0]
    prob_ij2 = ij.py.to_dataset(prob_np)

    return prob_ij2


def save_ij2(ij, image_ij2, outname):

    if os.path.exists(outname):
        os.remove(outname)

    ij.io().save(image_ij2, outname)


def get_outPlane_macro(filepath):
    return """
        open("%s");
        rename("outPlane.tif");
        mainWin = "outPlane.tif"

        setOption("BlackBackground", false);
        run("Make Binary", "method=Minimum background=Default calculate");
        run("Median 3D...", "x=4 y=4 z=4");
        run("Invert", "stack");
        run("Make Binary", "method=Minimum background=Default calculate");
        run("Dilate", "stack");
        run("Dilate", "stack");
        run("Median 3D...", "x=2 y=2 z=2");
        run("Invert", "stack");
    """ % (
        filepath
    )


def outPlane(filename):

    outPlane = sm.io.imread(f"datProcessing/{filename}/_outPlane{filename}.tif").astype(
        int
    )

    outPlane = ndimage.median_filter(outPlane, footprint=np.ones((10, 10, 10)))

    outPlane = np.asarray(outPlane, "uint8")
    tifffile.imwrite(f"datProcessing/{filename}/outPlane{filename}.tif", outPlane)


def woundsite(filename):

    wound = sm.io.imread(f"datProcessing/{filename}/outPlane{filename}.tif").astype(int)
    (T, X, Y) = wound.shape

    dfWoundDetails = pd.read_excel(f"datProcessing/WoundDetails.xlsx")
    dfWoundDetails["Wound centre Y"] = 512 - dfWoundDetails["Wound centre Y"]
    start = [0, 0]
    start[0] = int(
        dfWoundDetails["Wound centre X"][dfWoundDetails["Filename"] == filename]
    )
    start[1] = int(
        dfWoundDetails["Wound centre Y"][dfWoundDetails["Filename"] == filename]
    )

    vidLabels = []  # labels all wound and out of plane areas
    vidLabelsrc = []
    for t in range(T):
        img = sm.measure.label(wound[t], background=0, connectivity=1)
        imgxy = util.imgrcxy(img)
        vidLabelsrc.append(img)
        vidLabels.append(imgxy)

    _dfWound = []

    label = vidLabels[0][start[0], start[1]]
    contour = sm.measure.find_contours(vidLabels[0] == label, level=0)[0]
    poly = sm.measure.approximate_polygon(contour, tolerance=1)
    polygon = Polygon(poly)
    (Cx, Cy) = cell.centroid(polygon)
    wound[0][vidLabelsrc[0] != label] = 0
    mostLabel = label

    t = 0
    _dfWound.append(
        {
            "Time": t,
            "Polygon": polygon,
            "Position": (Cx, Cy),
            "Area": polygon.area,
            "Shape Factor": cell.shapeFactor(polygon),
        }
    )

    # Make a velocity dataframe
    dfVelocity = pd.read_pickle(
        f"datProcessing/{filename}/nucleusVelocity{filename}.pkl"
    )
    xf = Cx
    yf = Cy

    # track wound with time
    t = 0
    finished = False
    while t < T - 3 and finished != True:

        labels = vidLabels[t + 1][vidLabels[t] == mostLabel]

        uniqueLabels = set(list(labels))
        if 0 in uniqueLabels:
            uniqueLabels.remove(0)

        if len(uniqueLabels) == 0:
            finished = True
        else:
            count = Counter(labels)
            c = []
            for l in uniqueLabels:
                c.append(count[l])

            uniqueLabels = list(uniqueLabels)
            mostLabel = uniqueLabels[c.index(max(c))]
            C = max(c)

            if C < 50:
                finished = True
            else:
                contour = sm.measure.find_contours(
                    vidLabels[t + 1] == mostLabel, level=0
                )[0]
                poly = sm.measure.approximate_polygon(contour, tolerance=1)
                polygon = Polygon(poly)
                (Cx, Cy) = cell.centroid(polygon)
                wound[t + 1][vidLabelsrc[t + 1] != mostLabel] = 0

                t += 1

                _dfWound.append(
                    {
                        "Time": t,
                        "Polygon": polygon,
                        "Position": cell.centroid(polygon),
                        "Area": polygon.area,
                        "Shape Factor": cell.shapeFactor(polygon),
                    }
                )

    xf, yf = cell.centroid(polygon)
    tf = t + 1
    for t in range(tf, T - 1):

        x = [xf - 150, xf + 150]
        y = [yf - 150, yf + 150]
        dfxy = sortGrid(dfVelocity[dfVelocity["T"] == t], x, y)

        if len(dfxy) == 0:
            wound[t + 1] = 0
            [x, y] = [int(xf), int(512 - yf)]
            wound[t + 1][y - 2 : y + 2, x - 2 : x + 2] = 255
            _dfWound.append(
                {
                    "Time": t,
                    "Position": (xf, yf),
                }
            )
        else:
            v = np.mean(list(dfxy["Velocity"]), axis=0)

            xf = xf + v[0]
            yf = yf + v[1]

            wound[t + 1] = 0
            [x, y] = [int(xf), int(512 - yf)]
            wound[t + 1][y - 2 : y + 2, x - 2 : x + 2] = 255
            _dfWound.append(
                {
                    "Time": t,
                    "Position": (xf, yf),
                }
            )

    _dfWound.append(
        {
            "Time": t + 1,
            "Position": (xf, yf),
        }
    )

    dfWound = pd.DataFrame(_dfWound)
    dfWound.to_pickle(f"datProcessing/{filename}/woundsite{filename}.pkl")

    dist = []
    for t in range(T):
        img = 255 - util.imgrcxy(wound[t])
        dist.append(sp.ndimage.morphology.distance_transform_edt(img))

    dist = np.asarray(dist, "uint16")
    for t in range(len(dist)):
        img = imgxyrc(dist[t])
        dist[t] = img
    tifffile.imwrite(f"datProcessing/{filename}/distance{filename}.tif", dist)


def distance(filename):

    df = pd.read_pickle(f"datProcessing/{filename}/nucleusVelocity{filename}.pkl")
    mig = np.zeros(2)
    T = int(np.max(df["T"]))

    _df = []
    for t in range(T):
        dft = df[df["T"] == t]
        v = np.mean(dft["Velocity"])
        _df.append(
            {
                "Filename": filename,
                "T": t,
                "v": v,
            }
        )

    df = pd.DataFrame(_df)
    dist = []
    N = len(df)

    mig = np.array([256, 256])
    for t in range(N):

        img = np.zeros([512, 512])
        img[int(mig[0]), int(mig[1])] = 255
        img = 255 - img
        dist.append(sp.ndimage.morphology.distance_transform_edt(img))
        mig = mig + df["v"].iloc[t]

    dist = np.asarray(dist, "uint16")

    for t in range(len(dist)):
        img = imgxyrc(dist[t])
        dist[t] = img
    tifffile.imwrite(f"datProcessing/{filename}/distance{filename}.tif", dist)


def imgrcxy(img):

    n = len(img)

    imgxy = np.zeros(shape=(n, n))

    for x in range(n):
        for y in range(n):

            imgxy[x, y] = img[(n - 1) - y, x]

    return imgxy


def imgxyrc(imgxy):

    img = imgrcxy(imgxy)
    img = imgrcxy(img)
    img = imgrcxy(img)

    return img


def angle(filename):
    if "Wound" in filename:
        df = pd.read_pickle(f"datProcessing/{filename}/nucleusVelocity{filename}.pkl")
        T = int(np.max(df["T"]))
        df = pd.read_pickle(f"datProcessing/{filename}/woundsite{filename}.pkl")
        angle = np.zeros([T, 512, 512])

        y, x = np.mgrid[-512:512:1, -512:512:1]
        y = -y
        theta = np.arctan2(y, x) * 180 / np.pi
        theta = theta % 360

        mig = np.array([256, 256])
        for t in range(T):
            mig = np.array(df["Position"].iloc[t])
            if mig[0] < 0:
                mig[0] = 0
            img = theta[
                int(mig[1]) : (512 + int(mig[1])),
                512 - int(mig[0]) : (1024 - int(mig[0])),
            ]
            angle[t] = img

        angle = np.asarray(angle, "uint16")
        tifffile.imwrite(f"datProcessing/{filename}/angle{filename}.tif", angle)
    else:
        df = pd.read_pickle(f"datProcessing/{filename}/nucleusVelocity{filename}.pkl")
        mig = np.zeros(2)
        T = np.max(df["T"])

        _df = []
        for t in range(int(T)):
            dft = df[df["T"] == t]
            v = np.mean(dft["Velocity"])
            _df.append(
                {
                    "Filename": filename,
                    "T": t,
                    "v": v,
                }
            )

        df = pd.DataFrame(_df)
        N = len(df)
        angle = np.zeros([N, 512, 512])

        y, x = np.mgrid[-512:512:1, -512:512:1]
        y = -y
        theta = np.arctan2(y, x) * 180 / np.pi
        theta = theta % 360

        mig = np.array([256, 256])
        for t in range(N):

            img = theta[
                int(mig[1]) : (512 + int(mig[1])),
                512 - int(mig[0]) : (1024 - int(mig[0])),
            ]
            angle[t] = img
            mig = mig + df["v"].iloc[t]
            if mig[0] < 0:
                mig[0] = 0

        angle = np.asarray(angle, "uint16")
        tifffile.imwrite(f"datProcessing/{filename}/angle{filename}.tif", angle)


def deepLearningDiv(filename):

    print("Division Deep Learning Input")

    util.createFolder(f"datProcessing/dat_pred/{filename}")

    focus = sm.io.imread(f"datProcessing/{filename}/focus{filename}.tif").astype(int)
    focus = np.asarray(focus, "uint8")
    tifffile.imwrite(f"datProcessing/dat_pred/{filename}/focus{filename}.tif", focus)

    ecadFocus = focus[:, :, :, 1]
    h2Focus = focus[:, :, :, 0]

    (T, Y, X) = ecadFocus.shape

    inputVid = np.zeros([T - 4, 10, 512, 512])

    for i in range(5):
        inputVid[:, 0 + 2 * i] = ecadFocus[i : T - 4 + i]
        inputVid[:, 1 + 2 * i] = h2Focus[i : T - 4 + i]

    inputVid = np.asarray(inputVid, "uint8")
    tifffile.imwrite(
        f"datProcessing/dat_pred/{filename}/div10{filename}.tif",
        inputVid,
        imagej=True,
        metadata={"axes": "TZYX"},
    )


def deepLearningEcad(filename):

    print("E-cadherin Deep Learning Input")

    util.createFolder(f"datProcessing/dat_pred/{filename}")

    vid = sm.io.imread(f"datProcessing/{filename}/ecadHeight{filename}.tif").astype(int)
    T = vid.shape[0]
    seg = np.zeros([T, 512, 512, 3])
    seg[:, :, :, 0], seg[:, :, :, 1], seg[:, :, :, 2] = blurFocusStack(vid, 7)
    inputVid = np.zeros([T, 1024, 1024, 3])
    util.createFolder("image/")
    for t in range(T):
        image = seg[t]
        image = np.asarray(image, "uint8")
        tifffile.imwrite(f"image/image.tif", image)
        image = Image.open(f"image/image.tif")
        image = image.resize((1024, 1024))
        image = np.asarray(image, "uint8")
        inputVid[t] = image

    shutil.rmtree("image/")

    inputVid = np.asarray(inputVid, "uint8")
    tifffile.imwrite(
        f"datProcessing/dat_pred/{filename}/ecadBlur3{filename}.tif",
        inputVid,
    )


def deepLearningOutPlane(filename):

    print("Detecting Tissue Deep Learning Input")

    stack = sm.io.imread(f"datProcessing/{filename}/ecadHeight{filename}.tif").astype(
        int
    )

    T = stack.shape[0]
    inputVid = np.zeros([T, 512, 512, 3])
    inputVid[:, :, :, 0], inputVid[:, :, :, 1], inputVid[:, :, :, 2] = blurFocusStack(stack, 7)

    inputVid = np.asarray(inputVid, "uint8")
    tifffile.imwrite(
        f"datProcessing/dat_pred/{filename}/tissue{filename}.tif",
        inputVid,
    )


def blurFocusStack(vid, focusRange):

    vid = vid.astype("uint16")
    (T, Z, Y, X) = vid.shape
    variance = np.zeros([T, Z, Y, X])
    varianceMax = np.zeros([T, Y, X])
    surface = np.zeros([T, Y, X])
    focus = np.zeros([T, Y, X])
    focusDown = np.zeros([T, Y, X])
    focusUp = np.zeros([T, Y, X])

    for t in range(T):
        for z in range(Z):
            winMean = ndimage.uniform_filter(vid[t, z], (focusRange, focusRange))
            winSqrMean = ndimage.uniform_filter(
                vid[t, z] ** 2, (focusRange, focusRange)
            )
            variance[t, z] = winSqrMean - winMean ** 2

    varianceMax = np.max(variance, axis=1)

    for z in range(Z):
        surface[variance[:, z] == varianceMax] = z

    for z in range(Z):
        focus[surface == z] = vid[:, z][surface == z]

    for z in range(Z - 1):
        focusUp[surface == z] = vid[:, z + 1][surface == z]
    focusUp[surface == z + 1] = vid[:, z + 1][surface == z + 1]

    focusDown[surface == 0] = vid[:, 0][surface == 0]
    for z in range(1, Z):
        focusDown[surface == z] = vid[:, z - 1][surface == z]

    focusUp = focusUp.astype("uint8")
    focus = focus.astype("uint8")
    focusDown = focusDown.astype("uint8")

    focusUp, focus, focusDown = normaliseBlur(focusUp, focus, focusDown, 60)

    return focusUp, focus, focusDown


def normaliseBlur(focusUp, focus, focusDown, mu0):
    focus = focus.astype("float")
    (T, X, Y) = focus.shape

    for t in range(T):
        mu = focus[t, 50:450, 50:450][focus[t, 50:450, 50:450] > 0]

        mu = np.quantile(mu, 0.5)

        ratio = mu0 / mu

        focus[t] = focus[t] * ratio
        focus[t][focus[t] > 255] = 255

        focusUp[t] = focusUp[t] * ratio
        focusUp[t][focusUp[t] > 255] = 255

        focusDown[t] = focusDown[t] * ratio
        focusDown[t][focusDown[t] > 255] = 255

    return focusUp.astype("uint8"), focus.astype("uint8"), focusDown.astype("uint8")


def trackMate(filename):

    ### MAIN PROCESSING STEPS ###
    print("TrackMate")

    HyperStackDisplayer = sj.jimport(
        "fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer"
    )
    TmXmlReader = sj.jimport("fiji.plugin.trackmate.io.TmXmlReader")
    TmXmlWriter = sj.jimport("fiji.plugin.trackmate.io.TmXmlWriter")
    Logger = sj.jimport("fiji.plugin.trackmate.Logger")
    Settings = sj.jimport("fiji.plugin.trackmate.Settings")
    SelectionModel = sj.jimport("fiji.plugin.trackmate.SelectionModel")
    DetectorProvider = sj.jimport("fiji.plugin.trackmate.providers.DetectorProvider")
    TrackerProvider = sj.jimport("fiji.plugin.trackmate.providers.TrackerProvider")
    SpotAnalyzerProvider = sj.jimport(
        "fiji.plugin.trackmate.providers.SpotAnalyzerProvider"
    )
    EdgeAnalyzerProvider = sj.jimport(
        "fiji.plugin.trackmate.providers.EdgeAnalyzerProvider"
    )
    TrackAnalyzerProvider = sj.jimport(
        "fiji.plugin.trackmate.providers.TrackAnalyzerProvider"
    )
    jfile = sj.jimport("java.io.File")
    Model = sj.jimport("fiji.plugin.trackmate.Model")
    Trackmate = sj.jimport("fiji.plugin.trackmate.TrackMate")
    Factory = sj.jimport("fiji.plugin.trackmate.detection.LogDetectorFactory")
    LAPUtils = sj.jimport("fiji.plugin.trackmate.tracking.LAPUtils")
    SparseLAP = sj.jimport(
        "fiji.plugin.trackmate.tracking.sparselap.SimpleSparseLAPTrackerFactory"
    )
    FeatureFilter = sj.jimport("fiji.plugin.trackmate.features.FeatureFilter")
    ImagePlus = sj.jimport("ij.ImagePlus")

    stack_path = f"/Users/jt15004/Documents/Coding/python/processData/datProcessing/{filename}/migration{filename}.tif"
    xmlPath = (stack_path.rsplit(".tif"))[0]
    newXML = open(xmlPath + ".xml", "w")

    imp = ImagePlus(stack_path)
    model = Model()
    model.setLogger(Logger.IJ_LOGGER)
    settings = Settings()
    settings.setFrom(imp)
    settings.detectorFactory = Factory()
    #################### Change these settings based on trackmate parameters #####################################
    settings.detectorSettings = {
        "DO_SUBPIXEL_LOCALIZATION": True,
        "RADIUS": 9.000,
        "TARGET_CHANNEL": 1,
        "THRESHOLD": 10.000,
        "DO_MEDIAN_FILTERING": False,
    }
    # Configure tracker
    settings.trackerFactory = SparseLAP()
    settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap()
    settings.trackerSettings["LINKING_MAX_DISTANCE"] = 12.0
    settings.trackerSettings["GAP_CLOSING_MAX_DISTANCE"] = 12.0
    # Add ALL the feature analyzers known to TrackMate, via
    # providers.
    # They offer automatic analyzer detection, so all the
    # available feature analyzers will be added.
    spotAnalyzerProvider = SpotAnalyzerProvider()
    for key in spotAnalyzerProvider.getKeys():
        print(key)
        settings.addSpotAnalyzerFactory(spotAnalyzerProvider.getFactory(key))
    edgeAnalyzerProvider = EdgeAnalyzerProvider()
    for key in edgeAnalyzerProvider.getKeys():
        print(key)
        settings.addEdgeAnalyzer(edgeAnalyzerProvider.getFactory(key))
    trackAnalyzerProvider = TrackAnalyzerProvider()
    for key in trackAnalyzerProvider.getKeys():
        print(key)
        settings.addTrackAnalyzer(trackAnalyzerProvider.getFactory(key))

    trackmate = Trackmate(model, settings)
    # process
    ok = trackmate.checkInput()
    if not ok:
        print(str(trackmate.getErrorMessage()))
    ok = trackmate.process()
    if not ok:
        print(str(trackmate.getErrorMessage()))
    filter1 = FeatureFilter("TOTAL_INTENSITY", 83833.81, False)
    settings.addTrackFilter(filter1)
    jafile = jfile(xmlPath + ".xml")
    writer = TmXmlWriter(jafile)
    writer.appendModel(model)
    writer.appendSettings(settings)
    writer.writeToFile()
    imp.close()


def trackmate_vertices_import(trackmate_xml_path, get_tracks=False):
    """Import detected tracks with TrackMate Fiji plugin.

    ref: https://github.com/hadim/pytrackmate

    Parameters
    ----------
    trackmate_xml_path : str
        TrackMate XML file path.
    get_tracks : boolean
        Add tracks to label
    """

    root = et.fromstring(open(trackmate_xml_path).read())

    objects = []
    object_labels = {
        "FRAME": "t_stamp",
        "POSITION_T": "t",
        "POSITION_X": "x",
        "POSITION_Y": "y",
        "POSITION_Z": "z",
        "QUALITY": "q",
        "ID": "spot_id",
    }

    # features = root.find("Model").find("FeatureDeclarations").find("SpotFeatures")
    features = [
        "FRAME",
        "POSITION_T",
        "POSITION_X",
        "POSITION_Y",
        "POSITION_Z",
        "QUALITY",
        "ID",
    ]

    spots = root.find("Model").find("AllSpots")
    trajs = pd.DataFrame([])
    objects = []
    for frame in spots.findall("SpotsInFrame"):
        for spot in frame.findall("Spot"):
            single_object = []
            for label in features:
                single_object.append(spot.get(label))
            objects.append(single_object)

    trajs = pd.DataFrame(objects, columns=features)
    trajs = trajs.astype(float)

    # Apply initial filtering
    initial_filter = root.find("Settings").find("InitialSpotFilter")

    trajs = filter_spots(
        trajs,
        name=initial_filter.get("feature"),
        value=float(initial_filter.get("value")),
        isabove=True if initial_filter.get("isabove") == "true" else False,
    )

    # Apply filters
    spot_filters = root.find("Settings").find("SpotFilterCollection")

    for spot_filter in spot_filters.findall("Filter"):

        trajs = filter_spots(
            trajs,
            name=spot_filter.get("feature"),
            value=float(spot_filter.get("value")),
            isabove=True if spot_filter.get("isabove") == "true" else False,
        )

    trajs = trajs.loc[:, object_labels.keys()]
    trajs.columns = [object_labels[k] for k in object_labels.keys()]
    trajs["label"] = np.arange(trajs.shape[0])

    # Get tracks
    if get_tracks:
        filtered_track_ids = [
            int(track.get("TRACK_ID"))
            for track in root.find("Model").find("FilteredTracks").findall("TrackID")
        ]

        label_id = 0
        trajs["label"] = np.nan

        tracks = root.find("Model").find("AllTracks")
        for track in tracks.findall("Track"):

            track_id = int(track.get("TRACK_ID"))
            if track_id in filtered_track_ids:

                spot_ids = [
                    (
                        edge.get("SPOT_SOURCE_ID"),
                        edge.get("SPOT_TARGET_ID"),
                        edge.get("EDGE_TIME"),
                    )
                    for edge in track.findall("Edge")
                ]
                spot_ids = np.array(spot_ids).astype("float")[:, :2]
                spot_ids = set(spot_ids.flatten())

                trajs.loc[trajs["spot_id"].isin(spot_ids), "label"] = label_id
                label_id += 1

        # Label remaining columns
        single_track = trajs.loc[trajs["label"].isnull()]
        trajs.loc[trajs["label"].isnull(), "label"] = label_id + np.arange(
            0, len(single_track)
        )

    return trajs


def filter_spots(spots, name, value, isabove):
    if isabove:
        spots = spots[spots[name] > value]
    else:
        spots = spots[spots[name] < value]

    return spots


# -------------------


def nucleusVelocity(filename):

    # gather databases from tracking .xml file
    dfNucleus = trackmate_vertices_import(
        f"datProcessing/{filename}/migration{filename}.xml", get_tracks=True
    )

    uniqueLabel = list(set(dfNucleus["label"]))

    _dfTracks = []
    for label in uniqueLabel:

        # convert in to much simple dataframe
        df = dfNucleus.loc[lambda dfNucleus: dfNucleus["label"] == label, :]

        x = []
        y = []
        z = []
        t = []

        for i in range(len(df)):
            x.append(df.iloc[i, 2])
            y.append(511 - df.iloc[i, 3])  # this makes coords xy
            z.append(df.iloc[i, 4])
            t.append(df.iloc[i, 1])

        # fill in spot gaps in the tracks

        X = []
        Y = []
        Z = []
        T = []

        X.append(x[0])
        Y.append(y[0])
        Z.append(z[0])
        T.append(t[0])

        for i in range(len(df) - 1):
            t0 = t[i]
            t1 = t[i + 1]

            if t1 - t0 > 1:
                X.append((x[i] + x[i + 1]) / 2)
                Y.append((y[i] + y[i + 1]) / 2)
                Z.append((z[i] + z[i + 1]) / 2)
                T.append((t[i] + t[i + 1]) / 2)

            X.append(x[i + 1])
            Y.append(y[i + 1])
            Z.append(z[i + 1])
            T.append(t[i + 1])

        _dfTracks.append({"Label": label, "x": X, "y": Y, "z": Z, "t": T})

    df = pd.DataFrame(_dfTracks)

    _df2 = []
    for i in range(len(df)):
        t = df["t"][i]
        x = df["x"][i]
        y = df["y"][i]
        label = df["Label"][i]

        m = len(t)
        tMax = t[-1]

        if m > 1:
            for j in range(m - 1):
                t0 = t[j]
                x0 = x[j]
                y0 = y[j]

                x1 = x[j + 1]
                y1 = y[j + 1]

                v = np.array([x1 - x0, y1 - y0])

                _df2.append(
                    {
                        "Filename": filename,
                        "Label": label,
                        "T": t0,
                        "X": x0,
                        "Y": y0,
                        "Velocity": v,
                    }
                )

    dfVelocity = pd.DataFrame(_df2)

    dfVelocity.to_pickle(f"datProcessing/{filename}/nucleusVelocity{filename}.pkl")


def saveForSeg(filename):

    vid = sm.io.imread(f"datProcessing/{filename}/probEcad{filename}.tif").astype(int)
    if "Wound" in filename:
        dist = sm.io.imread(f"datProcessing/{filename}/distance{filename}.tif").astype(
            int
        )
        vid[dist == 0] = 0

    util.createFolder(f"datProcessing/{filename}/imagesForSeg")
    for t in range(len(vid)):
        if t > 99:
            T = f"{t}"
        elif t > 9:
            T = "0" + f"{t}"
        else:
            T = "00" + f"{t}"

        img = np.asarray(vid[t], "uint8")
        tifffile.imwrite(
            f"datProcessing/{filename}/imagesForSeg/ecadProb{filename}_{T}.tif", img
        )


def getBinary(filename):

    vid = sm.io.imread(f"datProcessing/{filename}/focus{filename}.tif").astype(float)
    T = len(vid)

    binary_vid = []
    track_vid = []
    T1_vid = []

    for frame in range(T):

        if frame < 10:
            framenum = f"00{frame}"
        elif frame < 100:
            framenum = f"0{frame}"
        else:
            framenum = f"{frame}"

        foldername = (
            f"datProcessing/{filename}/imagesForSeg/ecadProb{filename}_{framenum}"
        )

        imgRGB = sm.io.imread(foldername + "/handCorrection.tif").astype(float)
        img = imgRGB[:, :, 0]
        binary_vid.append(img)

        img = sm.io.imread(foldername + "/tracked_cells_resized.tif").astype(float)
        track_vid.append(img)

        img = sm.io.imread(foldername + "/stable_T1s.tif").astype(float)
        T1_vid.append(img)

    data = np.asarray(binary_vid, "uint8")
    tifffile.imwrite(f"datProcessing/{filename}/binary{filename}.tif", data)

    data = np.asarray(track_vid, "uint8")
    tifffile.imwrite(f"datProcessing/{filename}/binaryTracks{filename}.tif", data)

    data = np.asarray(T1_vid, "uint8")
    tifffile.imwrite(f"datProcessing/{filename}/T1s{filename}.tif", data)


def calShape(filename):

    binary = sm.io.imread(f"datProcessing/{filename}/binary{filename}.tif").astype(int)
    T = len(binary)
    _dfShape = []

    for t in range(T):

        img = binary[t]
        img = 255 - img
        imgxy = util.imgrcxy(img)

        # find and labels cells

        imgLabel = sm.measure.label(imgxy, background=0, connectivity=1)
        imgLabels = np.unique(imgLabel)[1:]
        allPolys = []
        allContours = []

        # converts cell boundary into a polygon

        for label in imgLabels:
            contour = sm.measure.find_contours(imgLabel == label, level=0)[0]
            poly = sm.measure.approximate_polygon(contour, tolerance=1)
            allContours.append(contour)
            allPolys.append(poly)

        allPolys, allContours = util.removeCells(allPolys, allContours)

        # quantifly polygon properties and saves them
        for i in range(len(allPolys)):

            poly = allPolys[i]
            contour = allContours[i]
            polygon = Polygon(poly)
            _dfShape.append(
                {
                    "Time": t,
                    "Polygon": polygon,
                    "Centroid": cell.centroid(polygon),
                    "Area": cell.area(polygon),
                    "Perimeter": cell.perimeter(polygon),
                    "Orientation": cell.orientation(polygon),
                    "Shape Factor": cell.shapeFactor(polygon),
                    "q": cell.qTensor(polygon),
                    "Trace(S)": cell.traceS(polygon),
                    "Polar": cell.polar(polygon),
                }
            )

    dfShape = pd.DataFrame(_dfShape)
    dfShape.to_pickle(f"datProcessing/{filename}/shape{filename}.pkl")


def nucleusShape(ij, filename, model_path):
    dfDivision = pd.read_pickle(f"datProcessing/{filename}/dfDivision{filename}.pkl")

    df = dfDivision[dfDivision["T"] > 10]
    df = df[df["X"] > 20]
    df = df[df["Y"] > 20]
    df = df[df["X"] < 492]
    df = df[df["Y"] < 492]

    if len(df) > 0:

        h2Stack = sm.io.imread(f"datProcessing/{filename}/{filename}.tif").astype(int)[
            :, :, 1
        ]
        h2 = sm.io.imread(f"datProcessing/{filename}/focus{filename}.tif").astype(int)[
            :, :, :, 0
        ]
        T, Z, X, Y = h2Stack.shape
        n = len(df)
        vidStackAll = np.zeros([11 * n, Z, 60, 60])
        vidAll = np.zeros([11 * n, 60, 60])

        for i in range(n):
            x = df["X"].iloc[i]
            y = 512 - df["Y"].iloc[i]
            t = int(df["T"].iloc[i])

            xMax = int(x + 30)
            xMin = int(x - 30)
            yMax = int(y + 30)
            yMin = int(y - 30)
            if xMax > 512:
                xMaxCrop = 60 - (xMax - 512)
                xMax = 512
            else:
                xMaxCrop = 60
            if xMin < 0:
                xMinCrop = -xMin
                xMin = 0
            else:
                xMinCrop = 0
            if yMax > 512:
                yMaxCrop = 60 - (yMax - 512)
                yMax = 512
            else:
                yMaxCrop = 60
            if yMin < 0:
                yMinCrop = -yMin
                yMin = 0
            else:
                yMinCrop = 0

            vidStack = np.zeros([10, Z, 60, 60])
            vid = np.zeros([10, 60, 60])
            for j in range(10):

                vidAll[11 * i + j, yMinCrop:yMaxCrop, xMinCrop:xMaxCrop] = h2[
                    t - 9 + j, yMin:yMax, xMin:xMax
                ]
                vidStackAll[
                    11 * i + j, :, yMinCrop:yMaxCrop, xMinCrop:xMaxCrop
                ] = h2Stack[t - 9 + j, :, yMin:yMax, xMin:xMax]

        vidAll = np.asarray(vidAll, "uint8")
        tifffile.imwrite(
            f"datProcessing/{filename}/vidH2{filename}.tif",
            vidAll,
            imagej=True,
            metadata={"axes": "TYX"},
        )

        vidStackAll = np.asarray(vidStackAll, "uint8")
        tifffile.imwrite(
            f"datProcessing/{filename}/vidStackH2{filename}.tif",
            vidStackAll,
            imagej=True,
            metadata={"axes": "TZYX"},
        )

        trackMateDiv(filename)

        stackProb = wekaNS(
            ij,
            filename,
            model_path,
        )
        stackProb = np.asarray(stackProb, "uint8")
        tifffile.imwrite(
            f"datProcessing/{filename}/nucleusProb{filename}.tif",
            stackProb,
            imagej=True,
            metadata={"axes": "TYX"},
        )

        stackProb[stackProb < 0.3 * 255] = 0
        stackProb[stackProb >= 0.3 * 255] = 255
        stackProb = np.asarray(stackProb, "uint8")
        tifffile.imwrite(
            f"datProcessing/{filename}/nucleusBinary{filename}.tif",
            stackProb,
            imagej=True,
            metadata={"axes": "TYX"},
        )

        dfNuclei = trackmate_vertices_import(
            f"datProcessing/{filename}/divisionNucleiTracks{filename}.xml",
            get_tracks=True,
        )
        dfDivision = pd.read_pickle(
            f"datProcessing/{filename}/dfDivision{filename}.pkl"
        )
        df = dfDivision[dfDivision["T"] > 10]
        df = df[df["X"] > 20]
        df = df[df["Y"] > 20]
        df = df[df["X"] < 492]
        df = df[df["Y"] < 492]
        n = len(df)

        _df = []

        for i in range(n):
            dft = dfNuclei[dfNuclei["t"] == i * 11 + 9]
            dft["R"] = 0
            dft["R"] = ((30 - dft["x"]) ** 2 + (30 - dft["y"]) ** 2) ** 0.5

            dfR = dft[dft["R"] < 12]
            if len(dfR) > 0:
                label = dfR["label"][dfR["z"] == np.min(dfR["z"])].iloc[0]

                df2 = dfNuclei[dfNuclei["label"] == label]

                if len(df2) == 10:
                    for j in range(10):
                        img = stackProb[i * 11 + j]
                        img = sp.ndimage.binary_fill_holes(img).astype(int) * 255
                        # img = np.asarray(img, "uint8")
                        # tifffile.imwrite(
                        #     f"dat/Unwound18h13/image/img.tif",
                        #     img,
                        # )
                        imgLabel = sm.measure.label(img, background=0, connectivity=1)
                        imgLabel = np.asarray(imgLabel, "uint8")
                        # tifffile.imwrite(
                        #     f"dat/Unwound18h13/image/imgLabel.tif",
                        #     imgLabel,
                        # )
                        x, y = df2["x"].iloc[j], df2["y"].iloc[j]
                        shapeLabel = imgLabel[int(y), int(x)]
                        if shapeLabel != 0:
                            # convert to row-col
                            imgLabel = util.imgrcxy(imgLabel)
                            contour = sm.measure.find_contours(
                                imgLabel == shapeLabel, level=0
                            )[0]
                            poly = sm.measure.approximate_polygon(contour, tolerance=1)
                            try:
                                polygon = Polygon(poly)
                                if j == 9:
                                    a = 0
                                _df.append(
                                    {
                                        "Filename": filename,
                                        "Div Label": df["Label"].iloc[i],
                                        "Div Orientation": df["Orientation"].iloc[i]
                                        % 180,
                                        "Track Label": label,
                                        "X": x,
                                        "Y": y,
                                        "Z": df2["z"].iloc[j],
                                        "T": df2["t"].iloc[j],
                                        "Polygon": polygon,
                                        "Area": cell.area(polygon) * scale ** 2,
                                        "Shape Orientation": (
                                            cell.orientation(polygon) * 180 / np.pi - 90
                                        )
                                        % 180,
                                        "Shape Factor": cell.shapeFactor(polygon),
                                        "q": cell.qTensor(polygon),
                                        "Time Before Division": j - 10,
                                    }
                                )
                            except:
                                print(i * 11 + j)
                                continue

        dfDivNucleus = pd.DataFrame(_df)

        dfDivNucleus.to_pickle(f"datProcessing/{filename}/dfDivNucleus{filename}.pkl")


def wekaNS(
    ij,
    filename,
    model_path,
):

    weka = sj.jimport("trainableSegmentation.WekaSegmentation")()
    weka.loadClassifier(model_path)
    stack = sm.io.imread(f"datProcessing/{filename}/vidH2{filename}.tif").astype(int)
    stack_ij2 = ij.py.to_dataset(stack)
    stackprob_ij2 = apply_weka(ij, weka, stack_ij2)

    return ij.py.from_java(stackprob_ij2).values


def trackMateDiv(filename):

    ### MAIN PROCESSING STEPS ###
    print("TrackMate")

    HyperStackDisplayer = sj.jimport(
        "fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer"
    )
    TmXmlReader = sj.jimport("fiji.plugin.trackmate.io.TmXmlReader")
    TmXmlWriter = sj.jimport("fiji.plugin.trackmate.io.TmXmlWriter")
    Logger = sj.jimport("fiji.plugin.trackmate.Logger")
    Settings = sj.jimport("fiji.plugin.trackmate.Settings")
    SelectionModel = sj.jimport("fiji.plugin.trackmate.SelectionModel")
    DetectorProvider = sj.jimport("fiji.plugin.trackmate.providers.DetectorProvider")
    TrackerProvider = sj.jimport("fiji.plugin.trackmate.providers.TrackerProvider")
    SpotAnalyzerProvider = sj.jimport(
        "fiji.plugin.trackmate.providers.SpotAnalyzerProvider"
    )
    EdgeAnalyzerProvider = sj.jimport(
        "fiji.plugin.trackmate.providers.EdgeAnalyzerProvider"
    )
    TrackAnalyzerProvider = sj.jimport(
        "fiji.plugin.trackmate.providers.TrackAnalyzerProvider"
    )
    jfile = sj.jimport("java.io.File")
    Model = sj.jimport("fiji.plugin.trackmate.Model")
    Trackmate = sj.jimport("fiji.plugin.trackmate.TrackMate")
    Factory = sj.jimport("fiji.plugin.trackmate.detection.LogDetectorFactory")
    LAPUtils = sj.jimport("fiji.plugin.trackmate.tracking.LAPUtils")
    SparseLAP = sj.jimport(
        "fiji.plugin.trackmate.tracking.sparselap.SimpleSparseLAPTrackerFactory"
    )
    FeatureFilter = sj.jimport("fiji.plugin.trackmate.features.FeatureFilter")
    ImagePlus = sj.jimport("ij.ImagePlus")

    stack_path = f"/Users/jt15004/Documents/Coding/python/processData/datProcessing/{filename}/vidStackH2{filename}.tif"
    xmlPath = f"/Users/jt15004/Documents/Coding/python/processData/datProcessing/{filename}/divisionNucleiTracks{filename}"
    newXML = open(xmlPath + ".xml", "w")

    imp = ImagePlus(stack_path)
    model = Model()
    model.setLogger(Logger.IJ_LOGGER)
    settings = Settings()
    settings.setFrom(imp)
    settings.detectorFactory = Factory()
    #################### Change these settings based on trackmate parameters #####################################
    settings.detectorSettings = {
        "DO_SUBPIXEL_LOCALIZATION": True,
        "RADIUS": 10.500,
        "TARGET_CHANNEL": 1,
        "THRESHOLD": 1.000,
        "DO_MEDIAN_FILTERING": False,
    }
    # Configure tracker
    settings.trackerFactory = SparseLAP()
    settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap()
    settings.trackerSettings["LINKING_MAX_DISTANCE"] = 12.000
    settings.trackerSettings["GAP_CLOSING_MAX_DISTANCE"] = 0.000
    # Add ALL the feature analyzers known to TrackMate, via
    # providers.
    # They offer automatic analyzer detection, so all the
    # available feature analyzers will be added.
    spotAnalyzerProvider = SpotAnalyzerProvider()
    for key in spotAnalyzerProvider.getKeys():
        print(key)
        settings.addSpotAnalyzerFactory(spotAnalyzerProvider.getFactory(key))
    edgeAnalyzerProvider = EdgeAnalyzerProvider()
    for key in edgeAnalyzerProvider.getKeys():
        print(key)
        settings.addEdgeAnalyzer(edgeAnalyzerProvider.getFactory(key))
    trackAnalyzerProvider = TrackAnalyzerProvider()
    for key in trackAnalyzerProvider.getKeys():
        print(key)
        settings.addTrackAnalyzer(trackAnalyzerProvider.getFactory(key))

    trackmate = Trackmate(model, settings)
    # process
    ok = trackmate.checkInput()
    if not ok:
        print(str(trackmate.getErrorMessage()))
    ok = trackmate.process()
    if not ok:
        print(str(trackmate.getErrorMessage()))
    filter1 = FeatureFilter("TOTAL_INTENSITY", 83833.81, False)
    settings.addTrackFilter(filter1)
    jafile = jfile(xmlPath + ".xml")
    writer = TmXmlWriter(jafile)
    writer.appendModel(model)
    writer.appendSettings(settings)
    writer.writeToFile()
    imp.close()