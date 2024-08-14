import cv2
import os
import numpy as np
from mtcnn import MTCNN

# Load saved recognition model
svm = cv2.ml.SVM_load("D:\CV_project\model.h5")

# Extract faces from images using MTCNN
def extract_faces(image_path, detector):
    img = cv2.imread(image_path)
    faces = []
    results = detector.detect_faces(img)
    for result in results:
        x, y, w, h = result['box']
        if h>100 and w>100:
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160))
            cv2.imwrite("D:\CV_project\ghgh.jpg", face)
            hog = cv2.HOGDescriptor()
            features = hog.compute(face).flatten()
            features=np.array(features, dtype=np.float32)
            features = np.expand_dims(features, axis=0)
            print(features.shape)
            result = svm.predict(features)
            #predicted_label = [k for k, v in label_dict.items() if v == int(result[1][0])]
            print(result)
            

    
    

# Load the MTCNN detector
detector = MTCNN()

img_path = 'D:\CV_project/Test4.jpg'

faces = extract_faces(img_path, detector)