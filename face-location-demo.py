#!/usr/bin/python
from PIL import Image, ImageDraw
import face_recognition
image = face_recognition.load_image_file("/home/lty/Pictures/two-members.jpg")
face_locations = face_recognition.face_locations(image, model="cnn")

print(face_locations)
print("I found {} face(s) in this photograph.".format(face_locations))

pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image, "RGBA")

for face_location in face_locations:

    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    face_range = [(left, top), (right, bottom)]
    d.line([(left, top), (right, top)], fill=(150, 0, 0, 128), width=5)
    d.line([(right, top), (right, bottom)], fill=(150, 0, 0, 128), width=5)
    d.line([(right, bottom), (left, bottom)], fill=(150, 0, 0, 128), width=5)
    d.line([(left, bottom), (left, top)], fill=(150, 0, 0, 128), width=5)
    # You can access the actual face itself like this:
    # face_image = image[top:bottom, left:right]
    # pil_image = Image.fromarray(face_image)

pil_image.show()
