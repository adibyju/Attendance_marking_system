import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from PIL import Image

# Enable eager execution
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# Define the paths to the training and testing data
train_path = "D:\CV_project\data"

# Define the image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32

# Create an instance of the ResNet-50 model with pre-trained weights
resnet = ResNet50(weights=None, include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the pre-trained weights so they are not updated during training
for layer in resnet.layers:
    layer.trainable = True

# Add a new fully-connected layer to the ResNet-50 model for classification
x = resnet.output
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(len(os.listdir(train_path)), activation="softmax")(x)

# Create a new model that includes the ResNet-50 and new classification layers
model = Model(inputs=resnet.input, outputs=predictions)

# Compile the model with a categorical cross-entropy loss function and an optimizer
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"], run_eagerly=True)

X_train = []
y_train = []
for i, person in enumerate(os.listdir(train_path)):
    person_path = os.path.join(train_path, person)
    for img_file in os.listdir(person_path):
        img_path = os.path.join(person_path, img_file)
        img = Image.open(img_path).resize((img_width, img_height))
        X_train.append(np.array(img))
        y_train.append(i)

# Convert the image and label lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Normalize the pixel values to be between 0 and 1
X_train = X_train / 255.0

# Convert the labels to one-hot encoded vectors
y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(os.listdir(train_path)))
print(len(os.listdir(train_path))) 

# Train the model on the resized images
model.fit(X_train, y_train, batch_size=batch_size, epochs=100)

# Save the trained model to a file
model.save("face_recognition_model.h5")

