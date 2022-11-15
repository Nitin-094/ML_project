import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime  

path = 'Image_DB'
images = []     
classNames = []  
myList = os.listdir(path)   #we can print a list of names of all the files present in the specified path.
print(myList)

# to find the no. unique images from directory
for i in myList:
    curImg = cv2.imread(f'{path}/{i}')
    images.append(curImg)
    classNames.append(os.path.splitext(i)[0]) # root & ext , returns a tuple.
print(classNames)

#If it finds , it will mark the attendence.
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()

        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            date = now.strftime("%d-%m-%Y")
            time =  now.strftime("%H:%M")
            f.writelines(f'\n{name},{date},{time}')
        

#to find the face encodings i.e features of the face , 66.
def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert BGR to RGB
        encode = face_recognition.face_encodings(img)[0] #For Given an image, return the 128-dimension face encoding for each face in the image
        encodeList.append(encode)
    #print(encodeList)
    return encodeList

encodeListKnown = findEncodings(images)



cap = cv2.VideoCapture(1) ## img capture


while True:
    success, img = cap.read()  # reading the captured image

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)#activate the resize feature to improve the processing speed.
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # resizing image acc to requirement & BGR image is converted to RGB.

    facesCurFrame = face_recognition.face_locations(imgS) #it detects a face from the frame . 
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame) # the two variables stores the data of the coming frame

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)  # matching and comparing the faces.
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  # Given a list of face encodings, compare them to a known face encoding and get a Euclidean distance for each comparison face. The distance tells you how similar the faces are.

        matchIndex = np.argmin(faceDis) #  to get the best fit , to get the best prob of the known face
#for pritning the name!
        if matches[matchIndex]:
            name = classNames[matchIndex]

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            if name in classNames:
                markAttendance(name)

    cv2.imshow('Webcam', img) #Displays an image in the separate window.
    key = cv2.waitKey(1) #Waits for a pressed key.
    if key == ord("q"):
        break
    
