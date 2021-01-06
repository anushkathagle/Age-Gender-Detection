#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2

video_capture = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    return(age_net, gender_net)

def video_detector(age_net,gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    while True:
        Temp, frame = video_capture.read()
        frame=cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3,5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, h:h + w].copy()   ## clear....
            blob=cv2.dnn.blobFromImage(face_img,1,(244,244),MODEL_MEAN_VALUES,swapRB=True)
            #blob=cv2.dnn.blobFromImage(face_img,1,(227,227),MODEL_MEAN_VALUES,swapRB=True)
            # Predict Gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            # Predict Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            overlay_text = "{} {}".format(gender, age)
            
            cv2.putText(frame, overlay_text, (x, y-15), font, 1, (255, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,255,0), 2)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) == 13:
            break
    video_capture.release()
    cv2.destroyAllWindows()


age_net, gender_net = load_caffe_models()  # load caffe models (age & gender)
video_detector(age_net, gender_net)  # prediction age & gender


# In[ ]:




