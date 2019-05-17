#!/usr/bin/python

from PIL import Image, ImageDraw
import face_recognition

image = face_recognition.load_image_file("resources/obama.jpg")

face_landmarks_list = face_recognition.face_landmarks(image)
# print("face lindmarks list:{}", face_landmarks_list)

pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image, "RGBA")

face_landmarks = face_landmarks_list[0]
# Make the eyebrows into a nightmare
# Print the location of each facial feature in this image
for facial_feature in face_landmarks.keys():
    print("The {} in this face has the following points: {}".
          format(facial_feature, face_landmarks[facial_feature]))

    # d.text(face_landmarks[facial_feature][0], facial_feature, fill=(150, 0, 0, 128), spacing=0, align="left" )

d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128),
          outline=(150, 0, 0, 256))
d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

# Gloss the lips
d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

# Sparkle the eyes
d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

# Apply some eyeliner
d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]],
       fill=(0, 0, 0, 110), width=6)
d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]],
       fill=(0, 0, 0, 110), width=6)

pil_image.show()
