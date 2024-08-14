import face_recognition
import cv2
import csv
import numpy as np
from datetime import datetime
import os
from mtcnn import MTCNN


# Get the video that is required from the live camera (now set to the camera of the laptop)
video_capture = cv2.VideoCapture(0)

# To run the same code for a provided video (if live video is not integrated)
# video_capture = cv2.VideoCapture("name.mp4")

# Loading the known faces
# Directory path where the images are stored
directory = "../../faces"

# Lists to store known face encodings and names
known_face_encodings = []
known_face_names = []

# Loop through the folders in the directory
for folder_name in os.listdir(directory):
    folder_path = os.path.join(directory, folder_name)
    if os.path.isdir(folder_path):
        # Loop through the images in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # Load the image file
            image = face_recognition.load_image_file(file_path)
            # Extract the face encoding
            encoding = face_recognition.face_encodings(image)[0]
            # Append the encoding and name to the lists
            known_face_encodings.append(encoding)
            known_face_names.append(folder_name)


# The students whose attendence is supposed to be marked and are expected to be present
students = known_face_names.copy()

face_locations = []
face_encodings = []

# Get the current date and time
now = datetime.now()
current_date = now.strftime("%d-%m-%y")

# Initialising the csv writer
f = open(f"../../results/{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

# The loop which checks through each of the frames in the video for identifying the faces
while True:

    _, frame = video_capture.read()
    # new_frame = cv2.resize(frame, (0,0), fx = 0.25, fy = 0.25)

    # Converting BRG to RGB
    rgb_new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Recognising faces
    face_locations = face_recognition.face_locations(rgb_new_frame)
    face_encodings = face_recognition.face_encodings(rgb_new_frame, face_locations)

    # Comparing the known faces to the faces now received
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # see if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding)

        name = "Unknown"

        # or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # draw a box around the face using OpenCV
        cv2.rectangle(frame, (left, top),
                      (right, bottom), (255, 0, 0), 2)

        # draw a label with a name below the face using OpenCV
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, 0.9, (255, 255, 255), 1)


        # update attendance in excel sheet
        if name in students:
            students.remove(name)
            current_time = now.strftime("%H:%M")
            lnwriter.writerow([name, current_time])

    cv2.imshow("",frame)

    # The break statement which stops the video feed by pressing 'q' on the keyboard
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()





