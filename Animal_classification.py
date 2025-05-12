import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import cv2
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data_dir = "/content/drive/MyDrive/Animal dataset"

class_names = ['Bear' , 'Bird' , 'Cat' , 'Cow' , 'Deer' , 'Dog' , 'Dolphin' , 'Elephant' , 'Giraffe' , 'Horse' , 'Kangaroo' , 'Lion' , 'Panda' , 'Tiger' , 'Zebra']
img_size = 224

data = []

for animal in class_names:
  class_path = os.path.join(data_dir , animal)
  label = class_names.index(animal)
  for img in os.listdir(class_path):
    img_path = os.path.join(class_path , img)

    img_arr = cv2.imread(img_path,0)
    img_arr = cv2.resize(img_arr , (img_size , img_size))

    data.append([img_arr , label])

random.shuffle(data)
x = []
y = []

for feature, label in data:
  x.append(feature)
  y.append(label)

pickle.dump(x,open('train_x.pkl','wb'))
pickle.dump(y,open('train_y.pkl','wb'))

x_train = pickle.load(open('train_x.pkl','rb'))
y_train = pickle.load(open('train_y.pkl','rb'))

x = np.array(x_train)
y = np.array(y_train)

x.max()

x = x/225.0

x.max()

print(img_size)

model = keras.Sequential(name="Sequential_1")
model.add(layers.Flatten(input_shape=(img_size,img_size), name="Flatten_layer"))
model.add(layers.Dense(224, activation='relu', name="layer_1"))
model.add(layers.Dense(224, activation='relu', name="layer_2"))
# Change the output layer to have 15 neurons (for 15 classes)
model.add(layers.Dense(15, activation='softmax', name="layer_3")) # Changed from 3 to 15

# Compile the model, use from_logits=False (default) because the output layer has a softmax activation
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), # from_logits is set to False (default)
              metrics=['accuracy'])

model.fit(x,y,epochs=20,validation_split=0.2)

Convnet Model

print(img_size)

modelc = keras.Sequential(name = "convModel") # Initialize modelc as a keras.Sequential object

modelc.add(layers.Conv2D(124 , (3,3) , activation = 'relu' , input_shape = (img_size , img_size, 1), name="conv1")) # Add input_shape
modelc.add(layers.MaxPooling2D((2,2) , name="maxpool1"))
modelc.add(layers.Dropout(0.2,name='dropout'),)

modelc.add(layers.Conv2D(64 , (3,3) , activation = 'relu' , name="conv2"))
modelc.add(layers.MaxPooling2D((2,2) , name="maxpool2"))

modelc.add(layers.Conv2D(32 , (3,3) , activation = 'relu' , name="conv3"))
modelc.add(layers.MaxPooling2D((2,2) , name="maxpool3"))
modelc.add(layers.Dropout(0.2,name='dropout3'),)

modelc.add(layers.Flatten(name="Flatten"))

modelc.add(layers.Dense(64 , activation = 'relu' , name="dense1"))
modelc.add(layers.Dense(32 , activation = 'relu' , name="dense2"))

modelc.add(layers.Dense(15 , activation = 'softmax' , name="outputDense")) # Changed from 3 to 15 to match number of classes

# Compile the model
modelc.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

modelc.summary()

history=modelc.fit(x,y,epochs=20,validation_split=0.2)

plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

img_size = 224
class_names = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']

new_image_path = '/content/drive/MyDrive/Animal dataset/Lion/Lion_11_4.jpg'
img = cv2.imread(new_image_path, 0)
img = cv2.resize(img, (img_size, img_size))
img = img / 255.0
img = img.reshape(1, img_size, img_size, 1)

prediction = modelc.predict(img)
predicted_class_index = np.argmax(prediction)
predicted_class_label = class_names[predicted_class_index]

print("Predicted Class:", predicted_class_label)
