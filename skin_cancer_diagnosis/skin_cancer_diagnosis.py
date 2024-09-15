# -*- coding: utf-8 -*-
"""
@author: Martin Mbalkam
skin cancer diagnosis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from sklearn.metrics import classification_report


#define the model
class CNN():
    
   def __init__(self):
        self.img_rows=150
        self.img_cols=150
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
       
  
   def create_cnn(self):
         
     # Define the CNN model
     model = Sequential()

     # Convolutional Block 1
     model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.img_shape))
     model.add(BatchNormalization())
     model.add(MaxPooling2D((2, 2)))

     # Convolutional Block 2
     model.add(Conv2D(64, (3, 3), activation='relu'))
     model.add(BatchNormalization())
     model.add(MaxPooling2D((2, 2)))

     # Convolutional Block 3
     model.add(Conv2D(128, (3, 3), activation='relu'))
     model.add(BatchNormalization())
     model.add(MaxPooling2D((2, 2)))

     # Convolutional Block 4
     model.add(Conv2D(256, (3, 3), activation='relu'))
     model.add(BatchNormalization())
     model.add(MaxPooling2D((2, 2)))

     # Flatten and Dense Layers
     model.add(Flatten())
     model.add(Dense(512, activation='relu'))
     model.add(Dropout(0.5))
     model.add(Dense(1, activation='sigmoid'))
      
     return model

   def train(self,path,epochs=50, batch_size=32):
       
       #data preprocessing for the model
       # Initialize ImageDataGenerators with validation_split
       train_datagen = ImageDataGenerator(
           rescale=1./255,
           rotation_range=20,
           width_shift_range=0.2,
           height_shift_range=0.2,
           shear_range=0.2,
           zoom_range=0.2,
           horizontal_flip=True,
           fill_mode='nearest',
           validation_split=0.2 
       )
       # Create data generators
       train_generator = train_datagen.flow_from_directory(
           path,
           target_size=(self.img_rows, self.img_cols),
           batch_size=batch_size,
           class_mode='binary',
           subset='training' 
       )

       validation_generator = train_datagen.flow_from_directory(
           path,
           target_size=(self.img_rows, self.img_cols),
           batch_size=batch_size,
           class_mode='binary',
           subset='validation'  
       )
       
       classifier=self.create_cnn()
       # Compile the model
       classifier.compile(loss='binary_crossentropy', 
                          optimizer='adam', metrics=['accuracy'])

       # Display the model's architecture
       classifier.summary()
       # Train the model
       history = classifier.fit(
           train_generator,
           steps_per_epoch=train_generator.samples // train_generator.batch_size,
           epochs=epochs,  
           validation_data=validation_generator,
           validation_steps=validation_generator.samples // validation_generator.batch_size
       )     
       #save model       
       classifier.save("skin_cancer_diagnosis.h5")
       
       #display graphic for accuracy
       plt.figure(figsize=(10, 10))

       plt.subplot(2, 2, 1)
       plt.plot(history.history['loss'], label='Loss')
       plt.plot(history.history['val_loss'], label='Validation Loss')
       plt.legend()
       plt.title('Training - Loss Function')

       plt.subplot(2, 2, 2)
       plt.plot(history.history['acc'], label='Accuracy')
       plt.plot(history.history['val_acc'], label='Validation Accuracy')
       plt.legend()
       plt.title('Train - Accuracy')

   def predict(self,x_test):
       
       model=Sequential()
       model.load("skin_cancer_diagnosis.h5")
       model.predict_classes(x_test)



if __name__ == '__main__':
    cnn = CNN()
    #change path with the path of the directory of your dataset
    cnn.train(path=r"c:\data\skin",epochs=80, batch_size=32)