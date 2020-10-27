# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:29:09 2020

@author: nyamochir
"""


import os
relative_path = 'Stanford40_JPEGImages/JPEGImages/'
#import shutil
import numpy as np
import cv2
# reason to set up on  224 is try to train on the prebuilt networks -->
w_size = 224
h_size = 224
classes = ['smoking', 'throwing_frisby', 'washing_dishes', 'blowing_bubbles', 'cleaning_the_floor', 'running', 'playing_violin', 'pouring_liquid', 'fishing', 'jumping', 'using_a_computer', 'riding_a_horse', 'writing_on_a_book', 'phoning', 'cutting_vegetables', 'pushing_a_cart', 'drinking', 'walking_the_dog', 'shooting_an_arrow', 'watching_TV', 'climbing', 'cutting_trees', 'feeding_a_horse', 'fixing_a_car', 'fixing_a_bike', 'waving_hands', 'playing_guitar', 'gardening', 'looking_through_a_telescope', 'reading', 'riding_a_bike', 'cooking', 'looking_through_a_microscope', 'applauding', 'brushing_teeth', 'rowing_a_boat', 'texting_message', 'writing_on_a_board', 'taking_photos', 'holding_an_umbrella']
classes = classes[:13]
classes.append("Not Sure")
#import matplotlib.pyplot as plt
from collections import Counter
#%matplotlib inline
#from PIL import Image
#import time
import model_with_13classes  as model
try:
    os.mkdir('realtime')
except:
    pass
def cleandir():
    if len(os.listdir('realtime'))!=0:
        for i in os.listdir('realtime'):
            os.remove('realtime/'+i)
    print('realtime dir cleaned')
def get_prediction(model, image, classes):
    w, l, d = (224, 224, 1)
    prediction = current_model.predict(image.reshape(1, w, l, d))
    #print(np.max(prediction), np.argmax(prediction), classes[np.argmax(prediction)])
    ret_val = np.max(prediction)
    if ret_val < 0.3:
        return classes[6]
    else:
        
        return classes[np.argmax(prediction)]
    
def initial_frames(model):
    """
    Never expect data to have o length
    
    """
    
    #lasses = ['applauding', 'phoning', 'jumping', 'reading', 'drinking', 'running', 'NotSure']
    #cleandir()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('frame', w_size, h_size)
    data = []
    #data = list(data)
    #data = data[1:]
    index = 0
    predicted_classes = []
    # sliding window with latest = 30 images -->  to respond or capture the results --> 
    text_to_display = "Current_Text"
    while True:
        ret, frame = cap.read()
        font = cv2.FONT_HERSHEY_SIMPLEX 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use putText() method for 
        # inserting text on video 
        
        #data = data[1:]
        
        if index != 0 and index %3 == 0:
            #cv2.imwrite('realtime/frame_'+str(index)+".jpg", frame)
            #file_ext = index-31
    
            #print(file_ext)
        #if len(os.listdir('realtime')) == 12:
        #        os.remove('realtime/'+os.listdir('realtime')[0])
            #time.sleep(1)
            np_frame = cv2.resize(frame, (w_size, h_size), interpolation = cv2.INTER_AREA)
            np_frame = np.asarray(np_frame)/255.# rescaling it
            data.append(np_frame)
        if len(data) == 16:
            
            if len(predicted_classes)==16:
                predicted_classes = predicted_classes[1:]
                predicted_classes.append(get_prediction(model, data[-1], classes))
            else:
                for image in data:
                    predicted_classes.append(get_prediction(model, image, classes))
            #print(data[0].shape)
            data  = data[1:]
        #print(np_frame.shape)
        # print(frame.)
        #data.append(np_frame)
        count_dic = Counter(predicted_classes)
        #print(len(count_dic), count_dic)
        cv2.putText(frame, 
                    text_to_display,  
                    (50, 50),  
                    font, 1,  
                    (0, 255, 255),  
                    2,  
                    cv2.LINE_AA) 
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
        text_to_display = str(count_dic)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):

            break
        index+=1
        
        
        
        
    cap.release()
    cv2.destroyAllWindows()
    data = np.array(data)
    print(data.shape)
    return data
current_model =   model.get_model()
current_model.load_weights("current_model_13_class_and_single_test_batch.h5")
data = initial_frames(current_model)