### IMPORTS ###
import configparser
import imagej
import os
import scyjava as sj
import functions
from os.path import exists

from pathlib import Path
import utils as util

filenames, fileType = util.getFilesType()

### MAIN PROCESSING STEPS ###
print("Processing image")

# Most of the arguments for this are being read from the GUI object (named "g").
# Furthermore, the numeric values are converted from strings (the GUI output)
# into integers.  The third argument controls the mu calculation during
# intensity normalisation.
filenames = ["PrewoundL18h10"]

for filename in filenames:
    print("-----------------------------------------------------------")
    print(f"{filename}")
    print("-----------------------------------------------------------")
    print("")

    path_to_file = f"datProcessing/{filename}/outPlane{filename}.tif"
    if False == exists(path_to_file):
        print("Out of Plane Zones")
        functions.outPlane(filename)

    if "Wound" in filename:
        path_to_file = f"datProcessing/{filename}/woundsite{filename}.pkl"
        if False == exists(path_to_file):
            print("Make Wound Database")
            functions.woundsite(filename)
    else:
        path_to_file = f"datProcessing/{filename}/distance{filename}.pkl"
        if False == exists(path_to_file):
            print("Make Distance")
            functions.distance(filename)

    path_to_file = f"datProcessing/{filename}/angle{filename}.pkl"
    if False == exists(path_to_file):
        print("Make Angle")
        functions.angle(filename)

    path_to_file = f"datProcessing/{filename}/imagesForSeg/ecadProb{filename}_000.tif"
    if False == exists(path_to_file):
        print("Save for Segmentation")
        functions.saveForSeg(filename)