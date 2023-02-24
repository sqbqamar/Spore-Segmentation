# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 12:10:09 2023

@author: Saqib Qamar
"""
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt


activation = 'relu'
feature_extractor = Sequential()
feature_extractor.add(Conv2D(32, 3, activation = activation, padding = 'same', input_shape = (SIZE_Y, SIZE_X, 3)))
feature_extractor.add(Conv2D(32, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
feature_extractor.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
#feature_extractor.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
feature_extractor.add(Conv2D(128, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(Conv2D(128, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(Conv2D(128, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
feature_extractor.add(Conv2D(164, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(Conv2D(164, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(Conv2D(164, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
feature_extractor.add(Conv2D(196, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(Conv2D(196, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(Conv2D(196, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))

model = feature_extractor
# Load pre-trained model and input image
#model = keras.applications.VGG16(weights='imagenet', include_top=True)
img_path = 'C:/Users/SAQIBQ/Desktop/Dataset/train/24.jpg'
img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
x = keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = keras.applications.vgg16.preprocess_input(x)

# Define function to extract layer activations
def get_layer_activations(model, layer_name, input_img):
    layer_output = model.get_layer(layer_name).output
    intermediate_model = keras.models.Model(inputs=model.input, outputs=layer_output)
    activations = intermediate_model.predict(input_img)
    return activations

# Get activations of a specific layer
layer_name = 'conv_layer1'
activations = get_layer_activations(model, layer_name, x)

# Extract the top 12 features from the activations
top_12_activations = activations[0,:,:,0:8]

# Visualize the top 12 features
fig, ax = plt.subplots(2, 4, figsize=(16, 8))

for i in range(2):
    for j in range(4):
        ax[i][j].imshow(activations[0, :, :, i*4+j], cmap='viridis', interpolation='nearest', extent=[0, 224, 224, 0])
        ax[i][j].axis('off')
        ax[i][j].set_title(f'Feature Map {i*4+j+1}', fontsize=10)

plt.tight_layout()
plt.savefig("C:/Users/SAQIBQ/Desktop/pred/24extractedblock2.jpg", dpi=700)
plt.show()
