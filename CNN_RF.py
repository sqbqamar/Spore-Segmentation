
@author: Saqib Qamar

Annotate images at www.apeer.com to create labels. 


import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import glob
import cv2
import joblib
from sklearn import metrics
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D




print(os.listdir("/pfs/stor10/users/home/s/sqbqamar/Public/Plant/Sphore/Dataset/"))

#Resizing images is optional, CNNs are ok with large images
SIZE_X = 2048 #Resize images (height  = X, width = Y)
SIZE_Y = 1668
SIZE_X1 = 128 #Resize images (height  = X, width = Y)
SIZE_Y1 = 104

#Capture training image info as a list
train_images = []

for directory_path in glob.glob("/pfs/stor10/users/home/s/sqbqamar/Public/Plant/Sphore/Dataset/train/"):
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        #train_labels.append(label)

#Convert list to array for machine learning processing        
train_images = np.array(train_images)

#Capture mask info as a list
train_masks = [] 
for directory_path in glob.glob("/pfs/stor10/users/home/s/sqbqamar/Public/Plant/Sphore/Dataset/label/"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tiff")):
        mask = cv2.imread(mask_path, 0)       
        mask = cv2.resize(mask, (SIZE_Y1, SIZE_X1))
        #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        train_masks.append(mask)
        #train_labels.append(label)
#Convert list to array for machine learning processing          
train_masks = np.array(train_masks)


#Use customary x_train and y_train variables
X_train = train_images
y_train = train_masks
y_train = np.expand_dims(y_train, axis=3) #May not be necessary


activation = 'relu'
feature_extractor = Sequential()
feature_extractor.add(Conv2D(32, 3, activation = activation, padding = 'same', input_shape = (SIZE_Y, SIZE_X, 3)))
#feature_extractor.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
feature_extractor.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
feature_extractor.add(Conv2D(128, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(Conv2D(128, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
#feature_extractor.add(Conv2D(128, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
feature_extractor.add(Conv2D(256, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(Conv2D(256, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(Conv2D(256, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
feature_extractor.add(Conv2D(512, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(Conv2D(512, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(Conv2D(512, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
#feature_extractor.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
feature_extractor.add(Conv2D(1024, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(Conv2D(1024, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(Conv2D(1024, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))


#features=feature_extractor.predict(X_train)
with tf.device('/cpu:0'):
    features=feature_extractor.predict(X_train)


#Plot features to view them

square = 8
ix=1
for _ in range(square):
    for _ in range(square):
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(features[0,:,:,ix-1], cmap='gray')
        ix +=1
plt.show()


#Reassign 'features' as X to make it easy to follow
X=features
X = X.reshape(-1, X.shape[3])  #Make it compatible for Random Forest and match Y labels

#Reshape Y to match X
Y = y_train.reshape(-1)

#Combine X and Y into a dataframe to make it easy to drop all rows with Y values 0
#In our labels Y values 0 = unlabeled pixels. 
dataset = pd.DataFrame(X)
dataset['Label'] = Y
print(dataset['Label'].unique())
print(dataset['Label'].value_counts())

##If we do not want to include pixels with value 0 
##e.g. Sometimes unlabeled pixels may be given a value 0 which is out of class. 
dataset = dataset[dataset['Label'] != 0]

#Redefine X and Y for Random Forest
X_for_RF = dataset.drop(labels = ['Label'], axis=1)
Y_for_RF = dataset['Label']

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 200, random_state = 42)

# Train the model on training data
model.fit(X_for_RF, Y_for_RF) 

#Save model for future use

#from sklearn.externals import joblib
filename = '/pfs/stor10/users/home/s/sqbqamar/Public/Plant/Sphore/VGGmodel_200.sav'
joblib.dump(model, filename)

#Load model.... 
loaded_model = joblib.load(filename)
#loaded_model = pickle.load(open(filename, 'rb'))

#from tensorflow.keras.models import load_model
#loaded_model = load_model('C:/Users/Kursadmin/Desktop/sphore/ML_DL_VGGmodel.sav')
#Test on a different image
#READ EXTERNAL IMAGE...

test_img = cv2.imread('/pfs/stor10/users/home/s/sqbqamar/Public/Plant/Sphore/Dataset/train/67.jpg', cv2.IMREAD_COLOR)  
#test_img = cv2.imread('C:/Users/Kursadmin/Desktop/sphore/Dataset/train/preprocessed/41.jpg', cv2.IMREAD_COLOR) 
test_img = cv2.resize(test_img, (SIZE_Y, SIZE_X))
test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
test_img = np.expand_dims(test_img, axis=0)

#predict_image = np.expand_dims(X_train[8,:,:,:], axis=0)
X_test_feature = feature_extractor.predict(test_img)
X_test_feature = X_test_feature.reshape(-1, X_test_feature.shape[3])

prediction = loaded_model.predict(X_test_feature)

#View and Save segmented image
prediction_image = prediction.reshape(mask.shape)
prediction_image = cv2.resize(prediction_image, (2048, 1668))
plt.imshow(prediction_image, cmap='jet')
cv2.imwrite('/pfs/stor10/users/home/s/sqbqamar/Public/Plant/Sphore/Prediction/pred_67.jpg', prediction_image)
#plt.imsave('C:/Users/Kursadmin/Desktop/sphore/Prediction/test_pred/test_pred_5.jpg', prediction_image, cmap='jet')


test_label = cv2.imread('/pfs/stor10/users/home/s/sqbqamar/Public/Plant/Sphore/Dataset/label/67.tiff', 0)
test_label = cv2.resize(test_label, (SIZE_Y1, SIZE_Y1))
test_label = test_label.reshape(-1)
print ("Accuracy on training data = ", metrics.accuracy_score(test_label, prediction)) # .9268 on image 46
list1 = metrics.accuracy_score(test_label, prediction)
list2 = []
list2.append(list1)
print(list2)
b = sum(list2)
print(b/64)
import pandas as pd
df = pd.DataFrame(list2)
print(df)
df.to_excel("/pfs/stor10/users/home/s/sqbqamar/Public/Plant/Sphore/Prediction/pred.xlsx")


