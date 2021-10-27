### IMPORTS ###
import configparser
import imagej
import os
import scyjava as sj
import utilBatch

from pathlib import Path


config_path = os.path.join(Path.home(), ".wbif", "autoseg.ini")

config = configparser.ConfigParser()

config.read(config_path)

if "defaults" in config:
    defaults = config["defaults"]
else:
    defaults = {}

f = open("pythonText.txt", "r")

filenames = f.read()
filenames = filenames.split(", ")
filename = filenames[0]

stack_path = f"/Users/jt15004/Documents/Coding/python/processData/datProcessing/{filename}/{filename}.tif"
ecad_model_path = "/Users/jt15004/Documents/Coding/classifiers/cellBoundary.model"
h2_model_path = "/Users/jt15004/Documents/Coding/classifiers/mitosis.model"
outOfPlane_model_path = "/Users/jt15004/Documents/Coding/classifiers/wound.model"

config["defaults"] = {
    "stack_path": stack_path,
    "ecad_model_path": ecad_model_path,
    "h2_model_path": h2_model_path,
    "outOfPlane_model_path": outOfPlane_model_path,
}


if not os.path.exists(config_path):
    os.makedirs(Path(config_path).parent)

# Writing the updated configuration to file
with open(config_path, "w") as configfile:
    config.write(configfile)


print("Initialising ImageJ (this may take a couple of minutes first time)")

# Setting the amount of RAM the Java Virtual Machine running ImageJ is allowed
# (e.g. Xmx6g loads 6 GB of RAM)
sj.config.add_option("-Xmx8g")
9
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

f = open("pythonText.txt", "r")

filenames = f.read()
filenames = filenames.split(", ")


for filename in filenames:
    print("-----------------------------------------------------------")
    print(f"{filename}")
    print("-----------------------------------------------------------")
    utilBatch.process_stack(
        ij,
        filename,
    )

    utilBatch.deepLearning(filename)
