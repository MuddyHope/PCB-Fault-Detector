#!/usr/bin/env python
# coding: utf-8


## CONFIG

from pathlib import Path
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image




# -------- AUTO BASE DIR (FIX FOR NOTEBOOK + SCRIPT) --------
def get_base_dir():
    # Jupyter notebook
    if "ipykernel" in os.sys.modules:
        base = Path.cwd()

        # walk up until Dataset folder is found
        while not (base / "Dataset").exists() and base != base.parent:
            base = base.parent

        return base

    # normal python script
    return Path(__file__).resolve().parent

BASE_DIR = get_base_dir()


# -------- CONFIG --------
DATASET_DIR = BASE_DIR / "Dataset"

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

MODEL_DIR = BASE_DIR

MODEL_PATH = str(MODEL_DIR / "model.h5")



# Load the model
def get_model():
    global model
    print(f"Path stored in {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("Model Loaded Successfully")



# In[3]:


def load_image(image_path):

    img = image.load_img(image_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    return img_tensor


# In[5]:


# Define the prediction model, and the prediction logic
def predict(img_path):
    new_image = load_image(img_path)
    get_model()
    pred = model.predict(new_image)

    print(pred)

    labels=np.array(pred)
    labels[labels>=0.6]=1
    labels[labels<0.6]=0

    print(labels)
    final=np.array(labels)

    if final[0][0]==1:
        return "Bad"
    else:
        return "Good"

