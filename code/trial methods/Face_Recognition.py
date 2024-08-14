import cv2
import os
import numpy as np


# Load labelled dataset
dataset_path = "D:\CV_project\Faces"
labels = os.listdir(dataset_path)
label_dict = {label: i for i, label in enumerate(labels)}
features = []
target = []
for label in labels:
    images_path = os.path.join(dataset_path, label)
    images = os.listdir(images_path)
    hog = cv2.HOGDescriptor()
    for image_name in images:
        image_path = os.path.join(images_path, image_name)
        image = cv2.imread(image_path)
        feature = hog.compute(image).flatten()
        features.append(feature)
        target.append(label_dict[label])

features = np.array(features, dtype=np.float32)
target = np.array(target, dtype=np.int32)
print(features.shape)
# Train recognition model
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(features, cv2.ml.ROW_SAMPLE, target)

# Save recognition model
svm.save("D:\CV_project\model.h5")

# Load saved recognition model
svm = cv2.ml.SVM_load("D:\CV_project\model.h5")

# Recognize faces in new images
new_image_path = "D:\CV_project\Faces\Aditya/7.jpg"
new_image = cv2.imread(new_image_path)
hog = cv2.HOGDescriptor()
features = hog.compute(new_image).flatten()
features=np.array(features, dtype=np.float32)
features = np.expand_dims(features, axis=0)
print(features.shape)
result = svm.predict(features)
predicted_label = [k for k, v in label_dict.items() if v == int(result[1][0])]
print(predicted_label)
