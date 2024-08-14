import os
import cv2
import numpy as np
from mtcnn import MTCNN

i=0

# Extract faces from images using MTCNN and store it in Faces directory
def extract_faces(image_path, detector, output_dir):
    img = cv2.imread(image_path)
    faces = []
    results = detector.detect_faces(img)
    for result in results:
        x, y, w, h = result['box']
        if h>100 and w>100:
            global i
            i=i+1
            face = img[y-50:y+h+50, x-50:x+w+50]
            output_path = os.path.join(output_dir, f"{i}.jpg")
            cv2.imwrite(output_path, face)

    
    

# Load the MTCNN detector
detector = MTCNN()

# Path to the folder containing the images
data_dir = '..\..\data'

# Create a folder to store the faces
output_dir = '../../faces'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each image in the input folder
for person_name in os.listdir(data_dir):
    person_path = os.path.join(data_dir, person_name)
    person_out = os.path.join(output_dir, person_name)
    if not os.path.exists(person_out):
        os.makedirs(person_out)
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        print(img_path)
        faces = extract_faces(img_path, detector, person_out)
    

