### IMPORTS ###
import configparser
import imagej
import os
import scyjava as sj
import utilBatch

from pathlib import Path
import commonLiberty as cl

config_path = os.path.join(Path.home(), ".wbif", "autoseg.ini")

config = configparser.ConfigParser()

config.read(config_path)

if "defaults" in config:
    defaults = config["defaults"]
else:
    defaults = {}

filenames, fileType = cl.getFilesType()
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
sj.config.add_option("-Xmx12g")

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

for filename in filenames:

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
        continue
    ok = trackmate.process()
    if not ok:
        print(str(trackmate.getErrorMessage()))
        continue
    filter1 = FeatureFilter("TOTAL_INTENSITY", 83833.81, False)
    settings.addTrackFilter(filter1)
    jafile = jfile(xmlPath + ".xml")
    writer = TmXmlWriter(jafile)
    writer.appendModel(model)
    writer.appendSettings(settings)
    writer.writeToFile()