#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras.models import load_model
from keras.preprocessing import image


# In[2]:


# Load the model
MODEL_PATH = "/Users/apple/Documents/CSUF/Spring 26/529/Circuit-Board-Fault-Detection-using-ML-main/model.h5"
def get_model():
    global model
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

