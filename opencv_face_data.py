import os
import PIL
import cv2
from PIL import Image
import numpy as np
from requests import patch

"""
數據訓練
"""

def getimageAndlabis(path): 
    #儲存人臉數據的二維數組
    faceSamples = []
    #儲存姓名數據
    ids = []
    #儲存圖片信息
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    #加載分類器
    face_detector = cv2.CascadeClassifier('face_detect.xml')
    #遍歷列表中的圖片
    for imagePath in imagePaths:
        #使用pil "L" 打開圖片並自動轉為灰階影像
        PIL_img = Image.open(imagePath).convert('L')
        #將圖片轉為數組
        img_numpy = np.array(PIL_img,'uint8')
        #獲取圖片的人臉特徵,保存為數組
        faces = face_detector.detectMultiScale(img_numpy)
        #獲取圖片的id
        id = int(os.path.split(imagePath)[1].split('.')[0])
        for x,y,w,h in faces:
            ids.append(id)
            faceSamples.append(img_numpy[y:y+h,x:x+w])
    print('id為',id)
    print('fs:',faceSamples)
    return faceSamples,ids

if __name__ =='__main__':
    #圖片路徑
    path="C:\\Users\\user\\Desktop\\opencv_img\\save\\"
    #獲取圖片數組和id數組與姓名
    faces,ids = getimageAndlabis(path)
    #加載識別器
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    #訓練
    recognizer.train(faces,np.array(ids))
    #保存文件
    recognizer.write(r'C:\Users\user\Desktop\python\trainer\trainer.yml')

    path="C:\\Users\\user\\Desktop\\opencv_img\\save2\\"
    faces1,ids = getimageAndlabis(path)
    recognizer1 =  cv2.face.LBPHFaceRecognizer_create()
    recognizer1.train(faces1,np.array(ids))
    recognizer1.write(r'C:\Users\user\Desktop\python\trainer\trainer1.yml')

    path="C:\\Users\\user\\Desktop\\opencv_img\\save3\\"
    faces2,ids = getimageAndlabis(path)
    recognizer2 =  cv2.face.LBPHFaceRecognizer_create()
    recognizer2.train(faces2,np.array(ids))
    recognizer2.write(r'C:\Users\user\Desktop\python\trainer\trainer2.yml')