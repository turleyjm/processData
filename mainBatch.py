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

ecad_model_path = "/Users/jt15004/Documents/Coding/classifiers/cellBoundary.model"
h2_model_path = "/Users/jt15004/Documents/Coding/classifiers/mitosis2.model"
outOfPlane_model_path = "/Users/jt15004/Documents/Coding/classifiers/wound.model"

config["defaults"] = {
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
    print(f"{filename}")
    print("-----------------------------------------------------------")
    utilBatch.process_stack(
        ij,
        filename,
    )
    utilBatch.deepLearning(filename)
    utilBatch.trackMate(filename)

for filename in filenames:
    print("-----------------------------------------------------------")
    print(f"{filename}")
    print("-----------------------------------------------------------")
    print("Running pixel classification (WEKA) Ecad")

    utilBatch.weka(
        ij,
        filename,
        ecad_model_path,
        "ecad",
        "ecadProb",
    )

    # print("Running pixel classification (WEKA) out of plane")

    # utilBatch.weka(
    #     ij,
    #     filename,
    #     outOfPlane_model_path,
    #     "ecad",
    #     "woundProb",
    # )

    # if "Wound" in filename:
    #     utilBatch.woundsite(ij, filename)


# At this point the analysis is complete
print("Complete")