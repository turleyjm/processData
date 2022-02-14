import os
import shutil
from math import floor, log10

from collections import Counter
import matplotlib
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scipy as sp
import scipy.linalg as linalg
from shapely.geometry import Polygon
import shapely
from shapely.geometry.polygon import LinearRing
import skimage as sm
import skimage.io
import skimage.measure
import tifffile
from skimage.draw import circle_perimeter
from scipy import optimize
import xml.etree.ElementTree as et


plt.rcParams.update({"font.size": 20})

# -------------------


def getFiles():
    f = open("pythonText.txt", "r")
    filenames = f.read()
    filenames = filenames.split(", ")
    return filenames


def getFilesType():
    f = open("pythonText.txt", "r")
    fileType = f.read()

    if fileType == "All":
        cwd = os.getcwd()
        Fullfilenames = os.listdir(cwd + "/datProcessing")
        filenames = []
        for filename in Fullfilenames:
            filenames.append(filename)

        if ".DS_Store" in filenames:
            filenames.remove(".DS_Store")
        if "woundDetails.xlsx" in filenames:
            filenames.remove("woundDetails.xlsx")
        if "dat_pred" in filenames:
            filenames.remove("dat_pred")
        if "confocalRawLocation.txt" in filenames:
            filenames.remove("confocalRawLocation.txt")

    else:
        cwd = os.getcwd()
        Fullfilenames = os.listdir(cwd + "/datProcessing")
        filenames = []
        for filename in Fullfilenames:
            if fileType in filename:
                filenames.append(filename)

    filenames.sort()

    return filenames, fileType


def getFilesOfType(fileType):

    cwd = os.getcwd()
    Fullfilenames = os.listdir(cwd + "/datProcessing")
    filenames = []
    for filename in Fullfilenames:
        if fileType in filename:
            filenames.append(filename)

    filenames.sort()

    return filenames


def ThreeD(a):
    lst = [[[] for col in range(a)] for col in range(a)]
    return lst


def sortTime(df, t):

    tMin = t[0]
    tMax = t[1]

    dftmin = df[df["Time"] >= tMin]
    df = dftmin[dftmin["Time"] < tMax]

    return df


def sortRadius(dfVelocity, t, r):

    rMin = r[0]
    rMax = r[1]
    tMin = t[0]
    tMax = t[1]

    dfrmin = dfVelocity[dfVelocity["R"] >= rMin]
    dfr = dfrmin[dfrmin["R"] < rMax]
    dftmin = dfr[dfr["Time"] >= tMin]
    df = dftmin[dftmin["Time"] < tMax]

    return df


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


def sortVolume(dfShape, x, y, t):

    xMin = x[0]
    xMax = x[1]
    yMin = y[0]
    yMax = y[1]
    tMin = t[0]
    tMax = t[1]

    dfxmin = dfShape[dfShape["X"] >= xMin]
    dfx = dfxmin[dfxmin["X"] < xMax]

    dfymin = dfx[dfx["Y"] >= yMin]
    dfy = dfymin[dfymin["Y"] < yMax]

    dftmin = dfy[dfy["T"] >= tMin]
    df = dftmin[dftmin["T"] < tMax]

    return df


def sortSection(dfVelocity, r, theta):

    rMin = r[0]
    rMax = r[1]
    thetaMin = theta[0]
    thetaMax = theta[1]

    dfxmin = dfVelocity[dfVelocity["R"] > rMin]
    dfx = dfxmin[dfxmin["R"] < rMax]

    dfymin = dfx[dfx["Theta"] > thetaMin]
    df = dfymin[dfymin["Theta"] < thetaMax]

    return df


def sortBand(dfRadial, band, pixelWidth):

    if band == 1:
        df = dfRadial[dfRadial["Wound Edge Distance"] < pixelWidth]
    else:
        df2 = dfRadial[dfRadial["Wound Edge Distance"] < band * pixelWidth]
        df = df2[df2["Wound Edge Distance"] >= (band - 1) * pixelWidth]

    return df


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


def rotation_matrix(theta):

    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    return R


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
    """
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def boundary(allPolys, allContours):

    _allPolys = []
    _allContours = []
    for i in range(len(allPolys)):
        poly = allPolys[i]
        polygon = Polygon(poly)
        pts = list(polygon.exterior.coords)

        test = vectorBoundary(pts)

        if test:
            _allPolys.append(poly)
            _allContours.append(allContours[i])

    allPolys = _allPolys
    allContours = _allContours

    return allPolys, allContours


def convex(allPolys, allContours):

    _allPolys = []
    _allContours = []
    for i in range(len(allPolys)):
        try:
            polys = allPolys[i]
            if notConvex(polys):
                _allPolys.append(allPolys[i])
                _allContours.append(allContours[i])
        except:
            continue

    allPolys = _allPolys
    allContours = _allContours
    return allPolys, allContours


def notConvex(polys):

    polygon = Polygon(polys)
    angleList = angles(polygon)
    m = len(angleList)
    for j in range(m):
        if angleList[j] > 4.5:
            return False
    return True


def angles(polygon):

    polygon = shapely.geometry.polygon.orient(polygon, sign=1.0)
    pts = list(polygon.exterior.coords)
    n = len(pts)
    pt = np.zeros(shape=(n + 1, 2))
    angles = []
    for i in range(n):
        pt[i][0] = pts[i][0]
        pt[i][1] = pts[i][1]
    pt[n] = pt[1]
    for i in range(n - 1):
        x = pt[i]
        y = pt[i + 1]
        z = pt[i + 2]
        u = x - y
        v = z - y
        theta = angle(u[0], u[1], v[0], v[1])
        angles.append(theta)
    return angles


def angle(u1, u2, v1, v2):

    top = u1 * v1 + u2 * v2
    bot = (u1 ** 2 + u2 ** 2) ** 0.5 * (v1 ** 2 + v2 ** 2) ** 0.5
    theta = np.arccos(top / bot)
    if u1 * v2 < u2 * v1:
        return theta
    else:
        return 2 * np.pi - theta


def inArea(polys, muArea, sdArea):
    polygon = Polygon(polys)
    area = polygon.area
    if area > muArea + 5 * sdArea:
        return False
    else:
        return True


def meanSdArea(allPolys):

    area = []
    n = len(allPolys)
    for i in range(n):
        polygon = Polygon(allPolys[i])
        area.append(polygon.area)
    muArea = np.mean(area)
    sdArea = np.std(area)
    return (muArea, sdArea)


def tooBig(allPolys, allContours):

    (muArea, sdArea) = meanSdArea(allPolys)
    _allPolys = []
    _allContours = []
    for i in range(len(allPolys)):
        try:
            polys = allPolys[i]
            if inArea(polys, muArea, sdArea):
                _allPolys.append(allPolys[i])
                _allContours.append(allContours[i])
        except:
            continue
    allPolys = _allPolys
    allContours = _allContours
    return allPolys, allContours


def removeCells(allPolys, allContours):

    allPolys, allContours = noArea(allPolys, allContours)
    allPolys, allContours = convex(allPolys, allContours)
    allPolys, allContours = tooBig(allPolys, allContours)
    allPolys, allContours = simple(allPolys, allContours)
    allPolys, allContours = boundary(allPolys, allContours)
    return allPolys, allContours


def simple(allPolys, allContours):

    _allPolys = []
    _allContours = []
    for i in range(len(allPolys)):
        try:
            polys = allPolys[i]
            if LinearRing(polys).is_simple:
                _allPolys.append(allPolys[i])
                _allContours.append(allContours[i])
        except:
            continue
    allPolys = _allPolys
    allContours = _allContours
    return allPolys, allContours


def noArea(allPolys, allContours):

    _allPolys = []
    _allContours = []
    for i in range(len(allPolys)):
        try:
            poly = allPolys[i]
            polygon = Polygon(poly)
            if polygon.area != 0:
                _allPolys.append(allPolys[i])
                _allContours.append(allContours[i])
        except:
            continue
    allPolys = _allPolys
    allContours = _allContours
    return allPolys, allContours


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


def imgrcxyRGB(vid):

    T, X, Y, C = vid.shape

    vidxy = np.zeros(shape=(T, X, Y, C))

    for x in range(X):
        for y in range(Y):

            vidxy[:, x, y] = vid[:, (Y - 1) - y, x]

    return vidxy


def imgxyrcRGB(vid):

    T, X, Y, C = vid.shape

    vidrc = np.zeros(shape=(T, X, Y, C))

    for x in range(X):
        for y in range(Y):

            vidrc[:, (Y - 1) - y, x] = vid[:, x, y]

    return vidrc


def vidrcxy(vid):

    T, X, Y = vid.shape

    vidxy = np.zeros(shape=(T, X, Y))

    for x in range(X):
        for y in range(Y):

            vidxy[:, x, y] = vid[:, (Y - 1) - y, x]

    return vidxy


def vidxyrc(vid):

    T, X, Y = vid.shape

    vidrc = np.zeros(shape=(T, X, Y))

    for x in range(X):
        for y in range(Y):

            vidrc[:, (Y - 1) - y, x] = vid[:, x, y]

    return vidrc


def imgxAxis(img):

    n = len(img)

    imgx = np.zeros(shape=(n, n))

    for x in range(n):
        for y in range(n):

            imgx[x, y] = img[(n - 1) - x, y]

    return imgx


def vectorBoundary(pts):

    n = len(pts)

    test = True

    for i in range(n):
        if pts[i][0] == 0 or pts[i][0] == 511:
            test = False
        elif pts[i][1] == 0 or pts[i][1] == 511:
            test = False
        else:
            continue
    return test