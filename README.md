# Circuit-Board-Fault-Defect-Detection-using-Image-Processing

Introduction : 

The project uses python programming language to identify the defects in a standard PCB using image processing. 
The model uses the MobileNets architechture, on top of which additional layers are added to the network to fine
tune the processing for this particular task. Most of the processing and training is done on our local computer and 
a very impressive accuracy rate has been achieved on the particular dataset it was tested on. It uses Jupyter 
Notebook to compile and execute the code.

The entire project contains three major files:
1. final_code.ipynb - Main code, to define and write code for the model
2. Prediction.ipynb - Prediction algorithm
3. Flask.ipynb - - Web based micro-server application to host the model online
   
-------------------------------------------------------------------------------------------------------------------

Installions Required :

1. Anaconda Navigator - To access Jupyter Notebook 
   However, the code can be accessed through any other editor as well
2. Python
3. Numpy : pip install numpy
4. Matplotlib : python -m pip install -U matplotlib
5. OpenCV : pip install opencv-python
6. Tensorflow : pip install tensorflow
   
-------------------------------------------------------------------------------------------------------------------

The code :

1. final_code.ipynb
   This code entitles the entire project, including the neural model used in the project and the training done for 
   the same. It generates a model and stores it. It also contains comments explaining each part of the code. 

2. Prediction.ipynb
   The file extracts the data from the already stored model, and uses this information to further process the 
   newly input data to predict if it is either a good or bad quality image.

3. Flask.ipynb
   Flask is a micro-server which is used to host a simple two webpage site on our local host computer. This file 
   contains the definition and deployement code for the server which runs the two html files which are stored in
   the templates folder in the project folder.
   
-------------------------------------------------------------------------------------------------------------------

Folders in the project :

1. Dataset :
   This folder consists of images from the database: https://github.com/tangsanli5201/DeepPCB
   This is a free dataset availble online which contains around 1100 images of PCB classified as Good and Bad.
   The size of the images has been reduced from 640*640 to 224*224 pixels, to help decrease the computional resources
   needed to train the model. This was atained by an anti-aliasing method, which helps in reducing image quality
   without major loss. All of the images are in binary format, i.e black and white where the black part indicated 
   the circuit components and the white parts show the empty space in the PCB layout.
   
   Download the dataset from the given link and save as a folder called 'Dataset' in your root directory.

3. Static :
   This folder contains the dataset of images used while testing using the Flask server. Altough the prediction on
   all these images are not accurate, it suffices the approximately 94% accuracy we have achieved. 

4. Templates :
   This folder contains the .html files which are rendered by the Flask server when deployed. It has a home page 
   and a predict page. Home page lets the user choose any image from their file system and the predict page displays
   the image choosen with the result for the same as either good or bad

5. model.h5 :
   This is the saved model after training the network, which is further used in both the files to predict the nature
   of the PCB, while testing.
   
-------------------------------------------------------------------------------------------------------------------

What is Flask Server? 

Flask is a micro web framework written in Python. It is classified as a microframework because it does not require
particular tools or libraries. It has no database abstraction layer, form validation, or any other components
where pre-existing third-party libraries provide common functions. However, Flask supports extensions that can add
application features as if they were implemented in Flask itself.

-------------------------------------------------------------------------------------------------------------------

Results:

![image](https://github.com/dhrupad-u/PCB-Defect-Detection-using-Machine-Learning/assets/42469685/f19cf229-fd09-4833-921c-dee0999af2b1)

![image](https://github.com/dhrupad-u/PCB-Defect-Detection-using-Machine-Learning/assets/42469685/4288b946-e2b0-45cb-a138-8b60aa4c5a79)
