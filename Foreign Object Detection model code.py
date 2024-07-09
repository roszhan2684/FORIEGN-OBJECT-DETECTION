#!/usr/bin/env python
# coding: utf-8

# In[1]:
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array
from keras.utils import load_img
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os


# In[8]:


# initialize the initial learning rate, number of epochs to train for,
# and batch size
# LR learning rate keeping low LR brings more accuracy
INIT_LR = 1e-4
EPOCHS = 20
# batch size
BS = 32

DIRECTORY = r"E:\Final Sem\Final year project\CD Content\Coding part\Datasets\data"
CATEGORIES = ["with_mask", "without_mask"]


# In[9]:


# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        
        data.append(image)
        labels.append(category)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)


if len(data) < 2:
    #print("Use the entire dataset for training")
    trainX = data
    trainY = labels
    testX = None
    testY = None
else:
    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# Check if testX and testY are empty
if testX is None or testY is None:
    print("Using the entire dataset for training.")
else:
    print("Data split into training and testing sets.")

    
testX = np.array(testX)
testY = np.array(testY)

aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))


headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)


# In[14]:


# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"], run_eagerly=True)

# train the head of the network



print("testX shape:", testX.shape)
print("testY shape:", testY.shape)


# In[24]:


# make predictions on the testing set
if testX is not None and testY is not None:
    print("[INFO] evaluating network...")
    predIdxs = model.predict(testX, batch_size=BS)

    # for each image in the testing set, find the index of the
    # label with the corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)

    # show a nicely formatted classification report
    print(classification_report(testY.argmax(axis=1), predIdxs,
        target_names=lb.classes_))
else:
    print("No test set defined. Skipping prediction step.")

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector1.model", save_format="h5")


# In[ ]:





# In[ ]:



