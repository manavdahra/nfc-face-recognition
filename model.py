import face_recognition
import cv2
import os
import constants
import PIL

def train():
    children = []
    encodings = []
    for root, _, files in os.walk(constants.DATASET_LOCATION):
        if len(files) > 0:
            print(root, files)
            file = files[0]
            # Load a sample picture and learn how to recognize it.
            try:
                child_img = face_recognition.load_image_file(root + '/' + file)
                child_face_encodings = face_recognition.face_encodings(child_img)
                child_id = os.path.basename(root)
                if len(child_face_encodings) > 0:
                    encodings.append(child_face_encodings[0])
                    children.append(child_id)
            except PIL.UnidentifiedImageError as e:
                print(e)
    print('Successfully trained model using images')
    return children, encodings
