#!/usr/bin/python
import numpy as np
import face_recognition
import cv2

# video writer build
video_path = "/home/lty/Pictures/ule-members-real.mp4"
input_movie = cv2.VideoCapture(video_path)
count = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))

print("Video:{} is {}*{} with frame number:{}".format(video_path, width, height, count))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
cv2.VideoWriter()
output_movie = cv2.VideoWriter('/tmp/output.avi', fourcc, 29.97, (width, height))

lty_image = face_recognition.load_image_file("/home/lty/Pictures/lty-face-source.png")
lty_encoding = face_recognition.face_encodings(lty_image)[0]

xxc_image = face_recognition.load_image_file("/home/lty/Pictures/xinxiucan.jpg")
xxc_encoding = face_recognition.face_encodings(xxc_image)[0]

yyy_image = face_recognition.load_image_file("/home/lty/Pictures/yangyuanyuan1.png")
yyy_encoding = face_recognition.face_encodings(yyy_image)[0]

zy_image = face_recognition.load_image_file("/home/lty/Pictures/zhangyi.jpg")
zy_encoding = face_recognition.face_encodings(zy_image)[0]

ces_image = face_recognition.load_image_file("/home/lty/Pictures/chenershuai.jpg")
ces_encoding = face_recognition.face_encodings(ces_image)[0]

known_faces = [
    lty_encoding,
    xxc_encoding,
    yyy_encoding,
    zy_encoding,
    ces_encoding
]

known_face_names = [
    "Luo",
    "Xin",
    "Yang",
    "Zhang",
    "Chen"
]

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
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame)
    face_names = [None]*len(face_encodings)
    # face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for index in range(len(face_encodings)):
        face_encoding = face_encodings[index]
        face_location = face_locations[index]
        # See if the face is a match for the known face(s)
        matchs = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.60)

        # If you had more than 2 faces, you could make this logic a lot prettier
        # but I kept it simple for the demo

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matchs[best_match_index]:
            face_names[index] = known_face_names[best_match_index]

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        if name:
            cv2.putText(frame, name, (left, top), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            print(matchs)
            print(name)

    output_movie.write(frame)
    print("完成度:{}/{}, 含有face数量:{}".format(frame_number, count, len(face_locations)))

# All done!
input_movie.release()
cv2.destroyAllWindows()

