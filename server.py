#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from utils import get_model


# In[2]:


app = Flask(__name__)


# In[3]:


# Load the Prediction model
def get_model():
    global model
    model = load_model('model.h5')
    print("Model loaded!")


# In[4]:


# Define the default image attributes, just like the one in the prediction file
def load_image(img_path):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor


# In[5]:


# Define the prediction model, and the prediction logic
def predict(img_path):
    new_image = load_image(img_path)

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


# In[6]:


get_model()


# In[7]:


# Folder to save uploads (inside static)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('home.html')


@app.route("/predict", methods=['GET', 'POST'])
def make_prediction():
    product = None
    file_path = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        # Save file safely
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        print("Saved:", file_path)

        # Call your prediction function
        product = predict(file_path)
        print("Prediction:", product)

    return render_template(
        'predict.html',
        product=product,
        user_image=file_path
    )


# In[9]:


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

