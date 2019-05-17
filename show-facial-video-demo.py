#!/usr/bin/python

import face_recognition
import cv2

video_path = "resources/short_hamilton_clip.mp4"
input_movie = cv2.VideoCapture(video_path)
count = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))

print("Video:{} is {}*{} with frame number:{}".format(video_path, width, height, count))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
cv2.VideoWriter()
output_movie = cv2.VideoWriter('/tmp/output.avi', fourcc, 29.97, (width, height))

face_locations = []
frame_number = 0

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file endsz
    if not ret:
        break

    # print(ret)
    # print(len(frame[0]))

    # Convert the image from BGR color (which OpenCV uses) to RGB color 
    # (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
    # face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left) in face_locations:
        # Draw a box around the face
        cv2.rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    output_movie.write(frame)
    print("完成度:{}/{}, 含有face数量:{}".format(frame_number, count, len(face_locations)))

# All done!
input_movie.release()
cv2.destroyAllWindows()
