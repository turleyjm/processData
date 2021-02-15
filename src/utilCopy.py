import numpy as np
import os
import scyjava as sj
import scipy
from scipy import ndimage

# This file contains the main processing functions.
#
# Note: ImageJ is currently undergoing a transformation behind the scenes.
# While the ImageJ GUI looks the same as it always has done, ImageJ2 uses a
# completely different method for storing images.  This format scales better
# with massive datasets (e.g. it can break them down into chunks) and works with
# more than 5 dimensions (the limit for ImageJ1).  As such, when working with
# PyImageJ, depending on what you're currently doing,  you may need to switch
# between ImageJ1, ImageJ2 and Numpy image formats.  The way this project worked
# out, I’ve had to use all three formats, so you'll see examples of each
# approach.

# This is the main image processing function.  It takes all the parameters we
# defined in the GUI and uses them to run the steps of the analysis - focusing,
# normalisation and WEKA.
def process_stack(
    ij,
    input_path,
    focus_range,
    model_path_Ecad,
    model_path_H2,
    save_focus,
    save_norm,
    save_prob,
):
    # Splitting the main filename from the file extension.  This is because we
    # will later on append information about the file being saved to the end of
    # the filename (e.g. "_Focused")
    (root_path, file_ext) = os.path.splitext(input_path)

    print("Finding Surface")

    ecadMacro = get_ecad_macro(root_path + file_ext)

    ij.script().run("macro.ijm", ecadMacro, True).get()

    ecad_ij2 = ij.py.active_dataset()
    ecad_np = ij.py.from_java(ecad_ij2)
    (T, Z, Y, X) = ecad_np.shape

    surface_np = getSurface(ecad_np)
    surface_ij2 = ij.py.to_dataset(surface_np)
    save = True
    if save:
        outname = "%s_surface.tif" % root_path
        save_ij2(ij, surface_ij2, outname)

    print("Filtering Height")

    heightFilter(ecad_np, surface_np)

    h2Macro = get_h2_macro(root_path + file_ext)
    ij.script().run("macro.ijm", h2Macro, True).get()

    h2_ij2 = ij.py.active_dataset()
    h2_np = ij.py.from_java(h2_ij2)

    heightFilter(h2_np, surface_np)

    save = False
    if save:
        outname = "%s_ecadHeight.tif" % root_path
        save_ij2(ij, ecad_ij2, outname)
        outname = "%s_h2Height.tif" % root_path
        save_ij2(ij, h2_ij2, outname)

    ### FOCUSING THE IMAGES ###
    # Running focus macro (using "get" on the end so the scipt waits until the
    # macro is complete)
    print("Focussing the image stack")

    # This function gets the macro text, which is stored in the get_focus_macro
    # function.  Since this macro loads an image, we need to update it to
    # hard-code the input image path.
    ecadFocus_np = focusStack(ij, ecad_np, 9)[1]

    # If selected in the GUI, this saves the focussed image to the same place as
    # the input file, but with the suffix "_Focussed".
    if save_focus:
        outname = "%s_ecadFocussed.tif" % root_path
        ecadfocus_ij2 = ij.py.to_dataset(ecadFocus_np)
        save_ij2(ij, ecadfocus_ij2, outname)

    # Focussing the image stack H2

    h2Focus_np = focusStack(ij, h2_np, 9)[1]

    if save_focus:
        outname = "%s_h2Focussed.tif" % root_path
        h2focus_ij2 = ij.py.to_dataset(h2Focus_np)
        save_ij2(ij, h2focus_ij2, outname)

    ### APPLYING NORMALISATION ###
    # For the normalisation process, we are doing processing in Python, so we
    # need to convert the ImageJ2 DefaultDataset that we got from the focussing
    # step into a Numpy array.  Fortunately, PyImageJ has some handy functions
    # which do this for us.
    print("Normalising images")

    # Ecad
    ecadfocus_np = ij.py.from_java(ecadfocus_ij2)

    # Running the normalise function defined later on in this file.  This will
    # update the input focus_np image.
    ecadfocus_np = normalise(ecadfocus_np.values, "MEDIAN", 30)
    ecadfocus_ij2 = ij.py.to_dataset(ecadfocus_np)

    # If selected in the GUI, this saves the normalised image to the same place
    # as the input file, but with the suffix "_Norm".  While we did the
    # processing for this step in Python, with the image as a Numpy array, the
    # ImageJ2 DefaultDataset and the Numpy array still correspond to the same
    # image in memory, so we can save using the ImageJ2 DefaultDataset.
    if save_norm:
        outname = "%s_eacdNorm.tif" % root_path
        save_ij2(ij, ecadfocus_ij2, outname)

    ### APPLYING PIXEL CLASSIFICATION (WEKA) ###
    # For WEKA pixel classification we go back to processing in ImageJ; however,
    # this time we can't run it as a simple macro.  This is a limitation of the
    # WEKA plugin, that it can't be run in headless mode as a macro.  Instead,
    # we use ScyJava's jimport to create an instance of the
    # WekaSegmentation Java class.  We're then able to use this class with all
    # it's associated functions.  By accessing the WekaSegmentation class
    # directly we can load in the .model classifier file and run classification
    # on a specific image.

    # H2

    h2focus_np = ij.py.from_java(h2focus_ij2)

    h2focus_np.values = normalise(h2focus_np.values, "UPPER_Q", 70)
    h2focus_ij2 = ij.py.to_dataset(h2focus_np)

    if save_norm:
        outname = "%s_h2Norm.tif" % root_path
        save_ij2(ij, h2focus_ij2, outname)

    print("Running pixel classification (WEKA) Ecad")

    weka = sj.jimport("trainableSegmentation.WekaSegmentation")()
    weka.loadClassifier(model_path_Ecad)

    # The apply_weka function takes our current ImageJ2 DefaultDataset object as
    # an input; however, it will convert it to ImageJ1 format when passing it to
    # WEKA - this is simply because the WekaSegmentation class hasn't been
    # designed to work with the newer ImageJ2 format.  The apply_weka function
    # outputs a new ImageJ2 DefaultDataset object containing the probability
    # maps.

    if T < 62:
        ecadprob_ij2 = apply_weka(ij, weka, ecadfocus_ij2)
    else:
        split = int(T / 61 - 1)
        stack_np = ecadfocus_np[0:61]
        stack_ij2 = ij.py.to_dataset(stack_np)
        stackprob_ij2 = apply_weka(ij, weka, stack_ij2)
        stackprob_np = ij.py.from_java(stackprob_ij2)

        for i in range(split):
            stack_np = ecadfocus_np[0 + 61 * (i + 1) : 61 + 61 * (i + 1)]
            stack_ij2 = ij.py.to_dataset(stack_np)
            stackprob_ij2 = apply_weka(ij, weka, stack_ij2)
            stackprob_npi = ij.py.from_java(stackprob_ij2)
            stackprob_np = np.concatenate(stackprob_np, stackprob_npi)

        stack_np = ecadfocus_np[0 + 61 * (i + 2) : -1]
        stack_ij2 = ij.py.to_dataset(stack_np)
        stackprob_ij2 = apply_weka(ij, weka, stack_ij2)
        stackprob_npi = ij.py.from_java(stackprob_ij2)
        stackprob_np = np.concatenate(stackprob_np, stackprob_npi)
        ecadprob_ij2 = ij.py.to_dataset(stack_np)

    print("Running pixel classification (WEKA) H2")

    weka = sj.jimport("trainableSegmentation.WekaSegmentation")()
    weka.loadClassifier(model_path_H2)

    if T < 62:
        h2prob_ij2 = apply_weka(ij, weka, h2focus_ij2)
    else:
        split = int(T / 61 - 1)
        stack_np = h2focus_np[0:61]
        stack_ij2 = ij.py.to_dataset(stack_np)
        stackprob_ij2 = apply_weka(ij, weka, stack_ij2)
        stackprob_np = ij.py.from_java(stackprob_ij2)

        for i in range(split):
            stack_np = h2focus_np[0 + 61 * (i + 1) : 61 + 61 * (i + 1)]
            stack_ij2 = ij.py.to_dataset(stack_np)
            stackprob_ij2 = apply_weka(ij, weka, stack_ij2)
            stackprob_npi = ij.py.from_java(stackprob_ij2)
            stackprob_np = np.concatenate(stackprob_np, stackprob_npi)

        stack_np = h2focus_np[0 + 61 * (i + 2) : -1]
        stack_ij2 = ij.py.to_dataset(stack_np)
        stackprob_ij2 = apply_weka(ij, weka, stack_ij2)
        stackprob_npi = ij.py.from_java(stackprob_ij2)
        stackprob_np = np.concatenate(stackprob_np, stackprob_npi)
        j2prob_ij2 = ij.py.to_dataset(stack_np)

    # If selected in the GUI, this saves the probability image to the same place
    # as the input file, but with the suffix "_Prob".
    if save_prob:
        outname = "%s_ecadProb.tif" % root_path
        save_ij2(ij, ecadprob_ij2, outname)
        outname = "%s_h2Prob.tif" % root_path
        save_ij2(ij, h2prob_ij2, outname)


# -----------------


def get_stack_macro(filepath):
    return """
        open("%s");
        main_win = getTitle();
        run("8-bit");
    """ % (
        filepath,
    )


def get_ecad_macro(filepath):
    return """
        open("%s");
        main_win = getTitle();
        run("Split Channels");
        selectWindow("C2-"+ main_win);
        close();
        selectWindow("C1-"+ main_win);
        run("8-bit");
    """ % (
        filepath,
    )


def get_h2_macro(filepath):
    return """
        open("%s");
        main_win = getTitle();
        run("Split Channels");
        selectWindow("C1-"+ main_win);
        close();
        selectWindow("C2-"+ main_win);
        run("Median 3D...", "x=2 y=2 z=2");
        run("8-bit");
    """ % (
        filepath,
    )


def getSurface(ecad_np):

    ecad_np = ecad_np.astype("float")
    variance = ecad_np
    surface = ecad_np[:, 0]
    (T, Z, Y, X) = ecad_np.shape

    for t in range(T):
        for z in range(Z):
            win_mean = ndimage.uniform_filter(ecad_np.values[t, z], (10, 10))
            win_sqr_mean = ndimage.uniform_filter(ecad_np.values[t, z] ** 2, (10, 10))
            variance[t, z] = win_sqr_mean - win_mean ** 2

    for t in range(T):
        variance[t] = ndimage.median_filter(variance[t], size=(3, 3, 3))

    peaks = np.zeros([T, Z - 1, Y, X])
    varDif = np.zeros([T, Z - 1, Y, X])
    varDif = variance[:, 0:-1].values - variance[:, 1:].values

    for z in range(Z - 1):
        peaks[:, z][varDif[:, z] > 0] = z
        peaks[:, z][varDif[:, z] <= 0] = Z

    for t in range(T):
        surface[t] = np.min(peaks[t], axis=0)

    for t in range(T):
        surface[t] = ndimage.median_filter(surface[t], (10, 10))
    surface = surface.astype("uint8")

    return surface


def heightScale(z0, z):

    # e where scaling starts from the surface and d is the cut off
    d = 10
    e = 8

    if z0 + e > z:
        scale = 1
    elif z > z0 + d:
        scale = 0
    else:
        scale = 1 - abs(z - z0 - e) / (d - e)

    return scale


def heightFilter(channel_np, surface_np):

    (T, Z, Y, X) = channel_np.shape

    for z in range(Z):
        for z0 in range(Z):
            scale = heightScale(z0, z)
            channel_np.values[:, z][surface_np.values == z0] = (
                channel_np.values[:, z][surface_np.values == z0] * scale
            )


# Returns the full macro code with the filepath and focus range inserted as
# hard-coded values.


def focusStack(ij, image_np, focus_range):

    image_np = image_np.astype("uint16")
    img = np.array(image_np)
    (T, Z, Y, X) = image_np.shape
    variance_np = np.zeros([T, Z, Y, X])
    varianceMax_np = np.zeros([T, Y, X])
    surface_np = np.zeros([T, Y, X])
    focus_np = np.zeros([T, Y, X])

    for t in range(T):
        for z in range(Z):
            win_mean = ndimage.uniform_filter(
                image_np.values[t, z], (focus_range, focus_range)
            )
            win_sqr_mean = ndimage.uniform_filter(
                image_np.values[t, z] ** 2, (focus_range, focus_range)
            )
            variance_np[t, z] = win_sqr_mean - win_mean ** 2

    for t in range(T):
        varianceMax_np[t] = np.max(variance_np[t], axis=0)

    for t in range(T):
        for z in range(Z):
            surface_np[t][variance_np[t, z] == varianceMax_np[t]] = z

    for t in range(T):
        for z in range(Z):
            focus_np[t][surface_np[t] == z] = img[t, z][surface_np[t] == z]

    surface_np = surface_np.astype("uint8")
    focus_np = focus_np.astype("uint8")

    return surface_np, focus_np


# Applies the normalisation code.  It's been reduced to a single copy (rather
# than having separate ones for Ecad and H2).
def normalise(vid, calc, mu0):
    vid = vid.astype("float")
    (T, X, Y) = vid.shape

    for t in range(T):
        mu = vid[
            t, 50:450, 50:450
        ]  # crop image to prevent skew in bleach from unbleached tissue

        # Since we're using the same function for Ecad and H2, we have this
        # conditional statement, which calculates mu appropriately depending
        # on which channel is currently being processed.
        if calc == "MEDIAN":
            mu = np.quantile(mu, 0.5)  # check this is right
        elif calc == "UPPER_Q":
            mu = np.quantile(mu, 0.75)

        ratio = mu0 / mu

        vid[t] = vid[t] * ratio
        vid[t][vid[t] > 255] = 255

    return vid.astype("uint8")


# Applies the WEKA pixel classification step.  This function creates an instance
# of the WekaSegmentation Java class, which allows us to run the class as if we
# were calling it natively inside Java.
def apply_weka(ij, classifier, image_ij2):
    # Using scyjava and jimport to create an instance of the ImageJFunctions
    # class, which will be used to convert ImageJ2 images to ImageJ1
    ijf = sj.jimport("net.imglib2.img.display.imagej.ImageJFunctions")()

    # Converting the ImageJ2 image (DefaultDataset) to an ImageJ1 (ImagePlus)
    # type.  The argument is just the name given to this image.  We can call it
    # anything we like.
    image_ij1 = ijf.wrap(image_ij2, sj.to_java("IJ1 image"))

    # Applying classifier using the WekaSegmentation class' "applyClassifier"
    # function.  This returns a new ImageJ1 image (ImagePlus format).
    prob_ij1 = classifier.applyClassifier(image_ij1, 6, True)

    # At the moment, the probability image is a single stack with alternating
    # predicted classes, so we want to convert it into a multidimensional stack.
    # To do this, we need to know how many frames and channels (classes) there
    # are.  The third dimension is labelled internally within the ImageJ1 image
    # as "channels", so to get the number of frames we actually need to find out
    # how many channels it has.
    n_channels = classifier.getNumOfClasses()
    n_frames = image_ij1.getNChannels()
    prob_ij1.setDimensions(n_channels, 1, n_frames)

    # Converting the probability image to ImageJ2 format, so it can be saved.
    return ij.py.to_dataset(prob_ij1)


# This function saves ImageJ2 format images to the specified output file path.
def save_ij2(ij, image_ij2, outname):
    # The image saving function will throw an error if it finds an image already
    # saved at the target location.  Therefore, we first need to delete such
    # images.
    if os.path.exists(outname):
        os.remove(outname)

    # Saving the ImageJ2 image (DefaultDataset) to file.
    ij.io().save(image_ij2, outname)
