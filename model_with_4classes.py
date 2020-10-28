# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:26:15 2020

@author: nyamochir
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 15:32:41 2020

@author: nyamochir
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import shutil
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will
#list all files under the input directory

import os
classes = ['applauding', 'phoning', 'jumping', 'cutting_trees']
 
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten

from tensorflow import random

import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
seed = 1
np.random.seed(seed)
random.set_seed(seed)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def makefs(dirs):
    try:
        os.mkdir('test')
    except:
        pass
    try:
        os.mkdir('training')
    except:
        pass
    for cdir in dirs:
        try:
            os.mkdir('test/'+cdir)
        except:
            pass
        try:
            os.mkdir('training/'+cdir)
        except:
            pass
def distribute_files():
    names = os.listdir('JPEGImages')
    path = 'JPEGImages/'
    dirs = []
    print("First_image_name ", names[0], " Length of the files: ",len(names))
    for name in names:
        current = name.split(".")[0][:-4]
        if current not in dirs:
            dirs.append(current)
    print(len(dirs),dirs)
    
    dirs = classes
    makefs(dirs)
    for cdir in dirs:
        index = 0
        for name in names:

            if cdir in name:
                if index > 15:

                    shutil.copy(path+name, 'training/'+cdir+"/"+name)
                else:
                    shutil.copy(path+name, 'test/'+cdir+"/"+name)

                index+=1
                
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
    model.add(Dense(4, activation= 'softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def train_model(model):
    train_datagen = ImageDataGenerator(rescale = 1./255., 
                                   rotation_range = 90,
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
    history = model.fit_generator(training_set, steps_per_epoch = step_epochs, epochs = 60, validation_data= test_set,validation_steps = len(test_set), shuffle = False)
    try:
        os.mkdir("weghts")
    except:
        pass
    model.save_weights('weights/current_model_4class_and_single_test_batch.h5')

    performance_model(history)
def performance_model(history):
    plt.plot(history.history['accuracy'], label = 'Training Accuracy')
    plt.plot(history.history['val_accuracy'], label = 'Validation accuracy')
    plt.legend()
    plt.show()
#distribute_files()