import os
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
    path_to_file = f"datProcessing/{filename}/binary{filename}.tif"
    if False == exists(path_to_file):
        print("Collecting binary images")
        utilBatch.getBinary(filename)

    path_to_file = f"datProcessing/{filename}/shape{filename}.pkl"
    if False == exists(path_to_file):
        print("Calculating cell shapes")
        utilBatch.calShape(filename)