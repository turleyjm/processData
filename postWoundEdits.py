### IMPORTS ###
import configparser
import imagej
import os
import scyjava as sj
import functions
from os.path import exists

from pathlib import Path
import utils as cl

filenames, fileType = cl.getFilesType()

for filename in filenames:
    print("-----------------------------------------------------------")
    print(f"{filename}")
    print("-----------------------------------------------------------")
    print("")

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