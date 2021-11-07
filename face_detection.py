import dlib
import face_recognition
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def detect_face(img):
    """
    :param img: the image read using the cv2.imread()
    :return: faces_array, which is a list of all detected faces.
             Each element of the array is a pair of a face area in gray and colored format
    """
    # convert the test image to gray image as OpenCV face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('xml_face_detection/haarcascade_frontalface_default.xml')

    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)

    faces_array = []
    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = picture[y:y+h, x:x+w]
        faces_array.append((roi_gray, roi_color))
        # cv2.rectangle(picture, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if len(faces_array) == 0:
        print("No faces detected")
        return None

    return picture, faces_array


if __name__ == "__main__":
    picture = cv2.imread("pictures/338.png")
    detection_result = detect_face(picture)
    if detection_result:
        picture_with_frame, all_faces = detection_result
        plt.imshow(cv2.cvtColor(all_faces[0][1], cv2.COLOR_BGR2RGB))
        plt.show()

