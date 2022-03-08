#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import the necessary packages
import os

# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
BASE_PATH = "dataset"

# define the names of the training, testing, and validation
# directories
TRAIN = "train"
TEST = "test"
VAL = "val"
AUG = "aug"

# initialize the list of class label names
CLASSES = ["Ae. aegypti", "Ae. melanimon", "An. coluzzii", "An. freeborni", "An. gambiae", "Cx. tarsalis"]

# set the batch size when fine-tuning
BATCH_SIZE = 16

# initialize the label encoder file path and the output directory to
# where the extracted features (in CSV file format) will be stored
LE_PATH = os.path.sep.join(["output", "le.cpickle"])
BASE_CSV_PATH = "output"

# set the path to the serialized model after training
MODEL_PATH = os.path.sep.join(["output", "ResNet50.model"])

# define the path to the output training history plots
UNFROZEN_PLOT_PATH = os.path.sep.join(["output", "unfrozen.png"])
WARMUP_PLOT_PATH = os.path.sep.join(["output", "warmup.png"])
