"""
Homework: Facial Recognition
Group Members: Ellie Gahan, Brett Kaliel, and Michael Rediron
Date: November 25, 2020
Course: CPSC 571
Filename: main.py
"""
import cv2
import sys
import os
from deepface import DeepFace
from multiprocessing import Process
import numpy
from tkinter import *
from tkinter import filedialog
import face_recognition
import time
from tkinter import simpledialog

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# opens a new window, so the new user can enter their name
def getUser2():
    # the input dialog
    USER_INP = simpledialog.askstring(title="New User",prompt="Please Enter your name:")

    return USER_INP

# this function does all the work from check who the user is to detecting the mood
def webcam():

    timer = True
    #setting timer 
    if timer:
        t = time.localtime(time.time())
        max_min = 1  + t.tm_min
        max_sec = t.tm_sec
        max_time = (60 * max_min) + max_sec
        timer = False
    
    #the font type use
    font = cv2.FONT_HERSHEY_DUPLEX

    #the lists
    knownFaceEncodings = []
    knownNames = []
    faceLocation = []

   #path to the fodler that holds the pictues of the users
    path = "./user pictures"

    # load all photos from a file and parse name
    for filename in os.listdir(path):

        if filename.endswith('.jpg') or filename.endswith('.png'):

            img = face_recognition.load_image_file(os.path.join(path, filename))  # load the image from file name
            name = filename[:-4]  # get the name of the image

            #encodes the img that it got from the file
            encoding = face_recognition.face_encodings(img)[0]
            #adds the name of the file to the list of users names
            knownNames.append(name)
            #adds the encodings to the encoding list
            knownFaceEncodings.append(encoding)

    # start camera and then compare photo to images stored
    name = ""
    video_capture = cv2.VideoCapture(0)
    cv2.namedWindow("Facial Recognition")
    userChecked = True

    while True:  # t   s while loop will check the users emotions

        if not video_capture.isOpened():
            video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            raise IOError("Cannot open webcam")

        ret, frame = video_capture.read()  # will read one image from a video]
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        frameImage = frame
        cv2.imshow("Facial Recognition", frame)

        #In this if statement it will identify which user is on the camera
        if userChecked:
            faceLocations = face_recognition.face_locations(rgb_small_frame)
            encoded = face_recognition.face_encodings(
                rgb_small_frame, faceLocations)

            
            for e in encoded:
                results = face_recognition.compare_faces(knownFaceEncodings, e)

                # need to get index from knownFaceEncoding
                # get the name of the person in the viedo
                if True in results:
                    firstResultIndex = results.index(True)
                    name = knownNames[firstResultIndex]
                else: #will do this else if the user face does not already exist in the knownFaceEncoding list
                    name = getUser2()
                    imageName = name + '.jpg'
                    # writing it to the folder  that holds users images
                    cv2.imwrite(os.path.join(path, imageName), frameImage)
               

        userChecked = False

        welcome = "Welcome " + name + "!"
        cv2.putText(frame, welcome, (30, 50), font, 1.0, (206 , 75, 27), 2)
        cv2.imshow("Facial Recognition", frame)

        #analyzes faces and returns an emotion
        result = DeepFace.analyze(small_frame, actions=['emotion'], models={}, enforce_detection=False)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangle around the faces
        for(x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

        #putting the emotion that was returned and writing it to the window with the camera
        cv2.putText(frame, result['dominant_emotion'],(30, 450), font, 2, (206 , 75, 27), 2, 1)
        cv2.imshow('Facial Recognition', frame)

        tt = time.localtime(time.time())
        t2_min = tt.tm_min
        t2_sec = tt.tm_sec
        end_time = (60 * t2_min) + t2_sec

        # this checks the time to see if the timer is up 
        if( end_time >= max_time ):
            
            frameBreak = frameImage
            cv2.putText(frameBreak, "Break Time !",(20, 395), font, 2, (255, 0, 0), 2, 1)
            cv2.imshow('Facial Recognition', frameBreak)
            timer = True

            #waits 10 secons before the window closes
            cv2.waitKey(10000)
            video_capture.release()
            cv2.destroyAllWindows()
            break

        #checkes to see if q is pressed and if it is then it quits and closes both windows         
        key = cv2.waitKey(3)
        if(key == ord('q') ): 
            video_capture.release()
            cv2.destroyAllWindows()
            root.destroy()
            break
   
#Create a root window
root = Tk()
root.title("Facial Recognition")
root.geometry("350x200")

# starts the webcam function
def startClick():
        webcam()
        return

# buttons 
start = Button(root,relief=RAISED, text="Run App",padx=50,pady=20, activeforeground="blue", font='times 14',command=startClick)
instruction1 = Label(root, text="Click Run App to start", font='times 14')
instruction2 = Label(root, text="If you are not in database please add name", font='times 14')
instruction3 =  Label(root, text="Time limit is set to 1 minute", font='times 14')
instruction4 =  Label(root, text="Please look at camera when starting app", font='times 14')

start.pack()
instruction1.pack()
instruction4.pack()
instruction2.pack()
instruction3.pack()

#Call the event loop
root.mainloop()
#Call the function main


