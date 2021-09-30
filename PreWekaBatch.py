### IMPORTS ###
import configparser
import imagej
import os
import scyjava as sj
import utilBatch

from pathlib import Path


f = open("pythonText.txt", "r")

filenames = f.read()
filenames = filenames.split(", ")


for filename in filenames:
    print("-----------------------------------------------------------")
    print(f"{filename}")
    print("-----------------------------------------------------------")
    utilBatch.process_stack(
        filename,
    )

    utilBatch.deepLearning(filename)
