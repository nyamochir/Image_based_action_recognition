# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

Created on Mon Oct  5 15:32:41 2020

@author: nyamochir
"""

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten



def data_generator():
    train_datagen = ImageDataGenerator(rescale = 1./255., 
                                   rotation_range = 45,
                                   fill_mode = 'nearest',
                                  shear_range = 0.2, 
                                  zoom_range = 0.2,
                                  horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255.)
    training_set = train_datagen.flow_from_directory('training',
                                                    target_size = (200, 200),
                                                    batch_size =80,
                                                    class_mode = 'categorical')
    test_set = test_datagen.flow_from_directory('test', 
                                               target_size = (200, 200), 
                                           batch_size = 1, 
                                           class_mode = 'categorical')
    step_epochs= len(training_set)
    return train_datagen, training_set, test_datagen, test_set, step_epochs

def get_model():
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = (200, 200, 3), padding = 'same', activation = 'relu'))
    model.add(Conv2D(32, (3, 3), activation ='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
    model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
    #model.add(Conv2D(64, 2, 2, activation =))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
    model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(2, 2))

    #model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
    #model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))

    #model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(6, activation= 'softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model
    