#!/usr/bin/python
import numpy as np
import face_recognition
from PIL import Image, ImageDraw, ImageFont

# lty_image = face_recognition.load_image_file("/home/lty/Pictures/lty-face-source.png")
# lty_encoding = face_recognition.face_encodings(lty_image)[0]

# xxc_image = face_recognition.load_image_file("/home/lty/Pictures/xinxiucan.jpg")
# xxc_encoding = face_recognition.face_encodings(xxc_image)[0]

# yyy_image = face_recognition.load_image_file("/home/lty/Pictures/yangyuanyuan.jpg")
# yyy_encoding = face_recognition.face_encodings(yyy_image)[0]

# zy_image = face_recognition.load_image_file("/home/lty/Pictures/zhangyi.jpg")
# zy_encoding = face_recognition.face_encodings(zy_image)[0]

# ces_image = face_recognition.load_image_file("/home/lty/Pictures/chenershuai.jpg")
# ces_encoding = face_recognition.face_encodings(ces_image)[0]

guochun_image = face_recognition.load_image_file("/home/lty/Pictures/guochun.jpg")
guochun_encoding = face_recognition.face_encodings(guochun_image)[0]

wangxudong_image = face_recognition.load_image_file("/home/lty/Pictures/wangxudong2.png")
wangxudong_encoding = face_recognition.face_encodings(wangxudong_image,)[0]

known_faces = [
    # lty_encoding,
    # xxc_encoding,
    # yyy_encoding,
    # zy_encoding,
    # ces_encoding
    guochun_encoding,
    wangxudong_encoding
]

known_face_names = [
    # "luo-tianyin",
    # "xin-xiucan",
    # "yang-yuanyuan",
    # "zhang-yi",
    # "chen-ershuai"
    "guo-chun",
    "wang-xudong"
]
font = ImageFont.load_default()

face_image_arr = face_recognition.load_image_file("/home/lty/Pictures/guochun.jpg")
face_image = Image.fromarray(face_image_arr)
face_image_draw = ImageDraw.Draw(face_image, "RGBA")
# face_image = face_recognition.load_image_file("resources/two_people.jpg")
face_locations = face_recognition.face_locations(face_image_arr)
face_encodings = face_recognition.face_encodings(face_image_arr)
face_names = [None]*len(face_encodings)

for index in range(len(face_encodings)):
    face_encoding = face_encodings[index]
    face_location = face_locations[index]
    # See if the face is a match for the known face(s)
    matchs = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

    # If you had more than 2 faces, you could make this logic a lot prettier
    # but I kept it simple for the demo

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_faces, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matchs[best_match_index]:
        face_names[index] = known_face_names[best_match_index]

# Label the results
for (top, right, bottom, left), name in zip(face_locations, face_names):
    face_image_draw.line([(left, top), (right, top)], fill=(150, 0, 0, 128), width=5)
    face_image_draw.line([(right, top), (right, bottom)], fill=(150, 0, 0, 128), width=5)
    face_image_draw.line([(right, bottom), (left, bottom)], fill=(150, 0, 0, 128), width=5)
    face_image_draw.line([(left, bottom), (left, top)], fill=(150, 0, 0, 128), width=5)

    if name:
        face_image_draw.text((left, top), name.encode('utf-8'))

print(face_names)
face_image.show()