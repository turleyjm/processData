### IMPORTS ###
import configparser
import imagej
import os
import scyjava as sj
import utilBatch
from os.path import exists

from pathlib import Path
import commonLiberty as cl

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
            utilBatch.woundsite(filename)

    path_to_file = f"datProcessing/{filename}/imagesForSeg/ecadProb{filename}_000.tif"
    if False == exists(path_to_file):
        print("Save for Segmentation")
        utilBatch.saveForSeg(filename)