import numpy as np
import os
import scyjava as sj
import scipy
from scipy import ndimage
import skimage as sm
import skimage.io
import tifffile


def process_stack(ij, filename):

    print("Finding Surface")

    stackFile = f"datProcessing/{filename}/{filename}.tif"
    stack = sm.io.imread(stackFile).astype(int)

    (T, Z, C, Y, X) = stack.shape

    surface = getSurface(stack[:, :, 0])
    if True:
        surface = np.asarray(surface, "uint8")
        tifffile.imwrite(f"datProcessing/{filename}/surface{filename}.tif", surface)

    print("Median Filter")

    for t in range(T):
        stack[t, :, 1] = ndimage.median_filter(stack[t, :, 1], size=(3, 3, 3))
        stack[t, :, 0] = ndimage.median_filter(stack[t, :, 0], size=(2, 2, 2))

    print("Filtering Height")

    ecad = heightFilter(stack[:, :, 0], surface)

    migration = np.asarray(stack[:, :, 1], "uint8")
    migration = normaliseMigration(migration, "MEDIAN", 10)

    tifffile.imwrite(f"datProcessing/{filename}/migration{filename}.tif", migration)

    h2 = heightFilter(stack[:, :, 1], surface)
    stack = 0

    if False:
        ecad = np.asarray(ecad, "uint8")
        tifffile.imwrite(f"datProcessing/{filename}/ecadHeight{filename}.tif", ecad)
        h2 = np.asarray(h2, "uint8")
        tifffile.imwrite(f"datProcessing/{filename}/h2Height{filename}.tif", h2)

    print("Focussing the image stack")

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

    ecadNormalise = normalise(ecadFocus, "MEDIAN", 60)
    h2Normalise = normalise(h2Focus, "UPPER_Q", 60)

    if True:
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


def weka(
    ij,
    filename,
    model_path,
    channel,
    name,
):

    framesMax = 7
    weka = sj.jimport("trainableSegmentation.WekaSegmentation")()
    weka.loadClassifier(model_path)

    ecadFile = f"datProcessing/{filename}/{channel}Focus{filename}.tif"
    ecad = sm.io.imread(ecadFile).astype(int)

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
        stackprob = np.zeros([T, X, Y])
        split = int(T / framesMax - 1)
        stack = ecad[0:framesMax]
        stack_ij2 = ij.py.to_dataset(stack)
        stackprob_ij2 = apply_weka(ij, weka, stack_ij2)
        stackprob[0:framesMax] = ij.py.from_java(stackprob_ij2).values

        j = 1
        print(f" part {j} -----------------------------------------------------------")
        j += 1

        for i in range(split):
            stack = ecad[framesMax * (i + 1) : framesMax + framesMax * (i + 1)]
            stack_ij2 = ij.py.to_dataset(stack)
            stackprob_ij2 = apply_weka(ij, weka, stack_ij2)
            stackprob[
                framesMax * (i + 1) : framesMax + framesMax * (i + 1)
            ] = ij.py.from_java(stackprob_ij2).values

            print(
                f" part {j} -----------------------------------------------------------"
            )
            j += 1

        stack = ecad[framesMax * (i + 2) :]
        stack_ij2 = ij.py.to_dataset(stack)
        stackprob_ij2 = apply_weka(ij, weka, stack_ij2)
        stackprob[framesMax * (i + 2) :] = ij.py.from_java(stackprob_ij2).values

        print(f" part {j} -----------------------------------------------------------")
        j += 1

    stackprob = 255 - np.asarray(stackprob, "uint8")
    tifffile.imwrite(f"datProcessing/{filename}/{name}{filename}.tif", stackprob)


# -----------------


def surfaceFind(p):

    n = len(p) - 4

    localMax = []
    for i in range(n):
        q = p[i : i + 5]
        localMax.append(max(q))

    Max = localMax[0]
    for i in range(n):
        if Max < localMax[i]:
            Max = localMax[i]
        elif Max < 250:
            continue
        else:
            return Max

    return Max


def getSurface(ecad):

    ecad = ecad.astype("float")
    variance = ecad
    (T, Z, Y, X) = ecad.shape

    for t in range(T):
        for z in range(Z):
            win_mean = ndimage.uniform_filter(ecad[t, z], (20, 20))
            win_sqr_mean = ndimage.uniform_filter(ecad[t, z] ** 2, (20, 20))
            variance[t, z] = win_sqr_mean - win_mean ** 2

    win_sqr_mean = 0
    win_mean = 0

    surface = np.zeros([T, X, Y])

    for t in range(T):
        for x in range(X):
            for y in range(Y):
                p = variance[t, :, x, y]

                # p = list(p)
                # p.reverse()
                # p = np.array(p)

                m = surfaceFind(p)
                h = [i for i, j in enumerate(p) if j == m][0]

                surface[t, x, y] = h

    surface = ndimage.median_filter(surface, size=9)
    surface = np.asarray(surface, "uint8")

    return surface


def heightScale(z0, z):

    # e where scaling starts from the surface and d is the cut off
    d = 15
    e = 14

    if z0 + e > z:
        scale = 1
    elif z > z0 + d:
        scale = 0
    else:
        scale = 1 - abs(z - z0 - e) / (d - e)

    return scale


def heightFilter(channel, surface):

    (T, Z, Y, X) = channel.shape

    for z in range(Z):
        for z0 in range(Z):
            scale = heightScale(z0, z)
            channel[:, z][surface == z0] = channel[:, z][surface == z0] * scale

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

    for t in range(T):
        varianceMax[t] = np.max(variance[t], axis=0)

    for t in range(T):
        for z in range(Z):
            surface[t][variance[t, z] == varianceMax[t]] = z

    for t in range(T):
        for z in range(Z):
            focus[t][surface[t] == z] = image[t, z][surface[t] == z]

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


def woundsite(ij, filename):

    filepath = f"/Users/jt15004/Documents/Coding/python/processData/datProcessing/{filename}/woundProb{filename}.tif"

    outPlaneMacro = get_outPlane_macro(filepath)
    ij.script().run("macro.ijm", outPlaneMacro, True).get()

    outPlaneBinary_ij2 = ij.py.active_dataset()
    outPlaneBinary = ij.py.from_java(outPlaneBinary_ij2)

    outPlaneBinary = np.asarray(outPlaneBinary, "uint8")
    tifffile.imwrite(f"datProcessing/{filename}/outPlane{filename}.tif", outPlaneBinary)


def deepLearning(filename):

    print("Deep Learning Input")

    ecadFocus = sm.io.imread(
        f"datProcessing/{filename}/ecadFocus{filename}.tif"
    ).astype(int)
    h2Focus = sm.io.imread(f"datProcessing/{filename}/h2Focus{filename}.tif").astype(
        int
    )
    (T, Y, X) = ecadFocus.shape

    input3h = np.zeros([T - 2, 512, 512, 3])
    input1e2h = np.zeros([T - 2, 512, 512, 3])

    for t in range(T - 2):
        input1e2h[t, :, :, 0] = h2Focus[t]
        input1e2h[t, :, :, 1] = ecadFocus[t]
        input1e2h[t, :, :, 2] = h2Focus[t + 1]

        input3h[t, :, :, 0] = h2Focus[t]
        input3h[t, :, :, 1] = h2Focus[t + 1]
        input3h[t, :, :, 2] = h2Focus[t + 2]

    input1e2h = np.asarray(input1e2h, "uint8")
    tifffile.imwrite(f"uploadDL/input1e2h{filename}.tif", input1e2h)
    input3h = np.asarray(input3h, "uint8")
    tifffile.imwrite(f"uploadDL/input3h{filename}.tif", input3h)


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
    print("done")

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
    settings.trackerSettings["LINKING_MAX_DISTANCE"] = 8.0
    settings.trackerSettings["GAP_CLOSING_MAX_DISTANCE"] = 8.0
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