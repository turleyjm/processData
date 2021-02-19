### IMPORTS ###
import configparser
import imagej
import os
import scyjava as sj
import utilEdit

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
h2_model_path = "/Users/jt15004/Documents/Coding/classifiers/mitosis2.model"
outOfPlane_model_path = "/Users/jt15004/Documents/Coding/classifiers/wound.model"
carryOn = True

config["defaults"] = {
    "stack_path": stack_path,
    "ecad_model_path": ecad_model_path,
    "h2_model_path": h2_model_path,
    "outOfPlane_model_path": outOfPlane_model_path,
    "carryOn": carryOn,
}


if not os.path.exists(config_path):
    os.makedirs(Path(config_path).parent)

# Writing the updated configuration to file
with open(config_path, "w") as configfile:
    config.write(configfile)


print("Initialising ImageJ (this may take a couple of minutes first time)")

# Setting the amount of RAM the Java Virtual Machine running ImageJ is allowed
# (e.g. Xmx6g loads 6 GB of RAM)
sj.config.add_option("-Xmx6g")

# Initialising PyImageJ with core ImageJ and the plugins we need.  For this, we
# have the Time_Lapse plugin, which offers an alternative for stack focusing.
# We also import the WEKA plugin.
ij = imagej.init(
    [
        "net.imagej:imagej:2.1.0",
        "net.imagej:imagej-legacy",
        "sc.fiji:Time_Lapse:2.1.1",
        "sc.fiji:Trainable_Segmentation:3.2.34",
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

carryOn = True
for filename in filenames:
    stack_path = f"/Users/jt15004/Documents/Coding/python/processData/datProcessing/{filename}/{filename}.tif"
    print("-----------------------------------------------------------")
    print(f"{filename}")
    print("-----------------------------------------------------------")
    utilEdit.process_stack(
        ij,
        stack_path,
        ecad_model_path,
        h2_model_path,
        outOfPlane_model_path,
        carryOn,
    )


# At this point the analysis is complete
print("Complete")