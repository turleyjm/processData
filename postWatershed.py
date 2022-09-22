### IMPORTS ###
import configparser
import imagej
import os
import scyjava as sj
import functions
from os.path import exists
from datetime import datetime

from pathlib import Path
import utils as util

filenames, fileType = util.getFilesType()
filename = filenames[0]

config_path = os.path.join(Path.home(), ".wbif", "autoseg.ini")

config = configparser.ConfigParser()

config.read(config_path)

if "defaults" in config:
    defaults = config["defaults"]
else:
    defaults = {}

stack_path = f"/Users/jt15004/Documents/Coding/python/processData/datProcessing/{filename}/{filename}.tif"
nucleusClassifer_path = (
    "/Users/jt15004/Documents/Coding/classifiers/nucleusClassifer.model"
)

config["defaults"] = {
    "stack_path": stack_path,
    "ecad_model_path": nucleusClassifer_path,
}


if not os.path.exists(config_path):
    os.makedirs(Path(config_path).parent)

# Writing the updated configuration to file
with open(config_path, "w") as configfile:
    config.write(configfile)


print("Initialising ImageJ (this may take a couple of minutes first time)")

# Setting the amount of RAM the Java Virtual Machine running ImageJ is allowed
# (e.g. Xmx6g loads 6 GB of RAM)
sj.config.add_option("-Xmx10g")

# Initialising PyImageJ with core ImageJ and the plugins we need.  For this, we
# have the Time_Lapse plugin, which offers an alternative for stack focusing.
# We also import the WEKA plugin.
ij = imagej.init(
    [
        "net.imagej:imagej:2.1.0",
        "net.imagej:imagej-legacy",
        "sc.fiji:Time_Lapse:2.1.1",
        "sc.fiji:Trainable_Segmentation:3.2.34",
        "sc.fiji:TrackMate_:5.1.0",
    ],
    headless=True,
)

# Displaying information about the running ImageJ
print(ij.getApp().getInfo(True))


### MAIN PROCESSING STEPS ###
print("Processing image")

# Most of the arguments for this are being read from the GUI object (named "g").
# Furthermore, the numeric values are converted from strings (the GUI output)
# into integers.  The third argument controls the mu calculation during
# intensity normalisation.

for filename in filenames:
    print("-----------------------------------------------------------")
    print(f"{filename}" + datetime.now().strftime(" %H:%M:%S"))
    print("-----------------------------------------------------------")
    print("")

    path_to_file = f"datProcessing/{filename}/T1s{filename}.tif"
    if False == exists(path_to_file):
        print("Collecting binary images")
        functions.getBinary(filename)

    path_to_file = f"datProcessing/{filename}/shape{filename}.pkl"
    if False == exists(path_to_file):
        print("Calculating cell shapes")
        functions.calShape(filename)

    path_to_file = f"datProcessing/{filename}/dfDivNucleus{filename}.pkl"
    if False == exists(path_to_file):
        print("Calculating dividing nuclei shapes")
        functions.nucleusShape(ij, filename, nucleusClassifer_path)

# At this point the analysis is complete
print("Complete")