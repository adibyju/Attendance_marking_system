import os
import imgaug.augmenters as iaa
import numpy as np
import cv2

train_path = "D:\CV_project\data"

# Define the augmentation pipeline
aug_pipeline = iaa.Sequential([
    iaa.Fliplr(p=0.5),  # horizontally flip 50% of the images
    iaa.Affine(rotate=(-25, 25)),  # rotate the image between -25 and 25 degrees
    iaa.Multiply((0.5, 1)),  # adjust brightness between 70% to 130% of original
    iaa.GaussianBlur(sigma=(0, 1.0))  # apply Gaussian blur with sigma ranging from 0 to 1.0
])

for i, person in enumerate(os.listdir(train_path)):
    person_path = os.path.join(train_path, person)
    for img_file in os.listdir(person_path):
        img_path = os.path.join(person_path, img_file)
        img = cv2.imread(img_path)
        for i in range(5):
            # Apply the augmentation pipeline to the image
            aug_img = aug_pipeline.augment_image(img)
            # Save the augmented image to the output directory
            cv2.imwrite(os.path.join('D:\CV_project','augmented',f'{person}_aug', f'{img_file[:-4]}_aug_{i}.jpg'), aug_img)