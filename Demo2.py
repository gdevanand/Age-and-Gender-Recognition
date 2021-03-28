import numpy
from pygame import mixer
import time
import cv2
import math
import sqlite3
from random import seed
from random import random
from tkinter import *
from PIL import ImageTk,Image
from tkinter import filedialog
import tkinter.messagebox
root=Tk()
root.geometry('500x570')
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH,expand=1)
root.title('Age and Gender Predictor')
frame.config(background='light blue')
label = Label(frame, text="Age and Gender Predictor",bg='light blue',font=('Times 30 bold'))
label.pack(side=TOP)
filename = PhotoImage(file="C:/Users/MARIAPPAN KARTHIK/Desktop/project/demo.png")
background_label = Label(frame,image=filename)
background_label.pack(side=TOP)


def data(age,gender,count):
    conn = sqlite3.connect('data.db')

    cursor = conn.cursor()

    with open('user'+str(count)+'.jpg','rb') as f:
        img=f.read()

    params = (age,gender,img)
    cursor.execute("INSERT INTO VALUE VALUES(?,?,?,datetime('now','localtime'))",params)
    conn.commit()
    cursor.close()
    conn.close()


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes





def exitt():
   exit()


def image():
    root.filename = filedialog.askopenfilename(initialdir="/project", title="select image",filetypes=(("jpg files", "*.jpg"),("all files","*.*")))
    my_label = Label(root, text=root.filename).pack()
    my_image = ImageTk.PhotoImage(Image.open(root.filename))
    my_image_label = Label(image=my_image).pack()
    
  
def web():
   capture =cv2.VideoCapture(0)
   while True:
      ret,frame=capture.read()
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      cv2.imshow('frame',frame)
      if cv2.waitKey(1) & 0xFF ==ord('q'):
         break
   capture.release()
   cv2.destroyAllWindows()

def webdet():
    faceProto="opencv_face_detector.pbtxt"
    faceModel="opencv_face_detector_uint8.pb"
    ageProto="age_deploy.prototxt"
    ageModel="age_net.caffemodel"
    genderProto="gender_deploy.prototxt"
    genderModel="gender_net.caffemodel"

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList=['(0-3)', '(4-6)', '(8-12)','(13-15)', '(18-21)','(22-25)','(26-30)','(31-35)', '(36-40)', '(41-46)', '(47-55)', '(56-64)', '(65-100)']
    genderList=['Male','Female']
    seed(1)
    count=random()

    faceNet=cv2.dnn.readNet(faceModel,faceProto)
    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)

    video=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    padding=20
    while(video.isOpened()):
        hasFrame, frame=video.read()
        if not hasFrame:
            cv2.waitKey()
            break

        resultImg,faceBoxes=highlightFace(faceNet,frame)
        #if not faceBoxes:
            #print("No face detected")

        for faceBox in faceBoxes:
            face=frame[max(0,faceBox[1]-padding):
                       min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                       :min(faceBox[2]+padding, frame.shape[1]-1)]

            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            
            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]
            

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
            cv2.imshow("Detecting age and gender", resultImg)
            



        if cv2.waitKey(1) & 0xFF == ord('s'):
            
            print(f'Gender: {gender}')
            print(f'Age: {age[1:-1]} years')
            cv2.imwrite('user'+str(count)+'.jpg',resultImg)
            cv2.rectangle(resultImg,(0,200),(640,300),(0,255,0),cv2.FILLED)
            cv2.putText(resultImg,"Face Saved",(150,265),cv2.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
            cv2.imshow("Detecting age and gender",resultImg)
            cv2.waitKey(0)

            data(f'{age}',f'{gender}',count)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(1) & 0xFF == ord('e'):
            break
            video.release()
            cv2.destroyAllWindows()

but1=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=web,text='Open Cam',font=('helvetica 15 bold'))
but1.place(x=5,y=176)

#but2=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=image,text='Open Cam & Record',font=('helvetica 15 bold'))
#but2.place(x=5,y=176)

but3=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=webdet,text='Open Cam & Detect',font=('helvetica 15 bold'))
but3.place(x=5,y=322)

but5=Button(frame,padx=5,pady=5,width=5,bg='white',fg='black',relief=GROOVE,text='EXIT',command=exitt,font=('helvetica 15 bold'))
but5.place(x=210,y=478)

root.mainloop()