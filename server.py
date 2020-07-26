import face_recognition
import cv2
import numpy as np
import os
import io
import zlib
import data
import model
from flask import Flask
from flask import request

children = []
encodings = []
face_locations = []
face_encodings = []
face_names = []

app = Flask(__name__)

def init():

    children_exist = os.path.exists('./children.npy')
    encodgins_exist = os.path.exists('./encodings.npy')
    print('{} {}', children_exist, encodgins_exist)

    if not os.path.exists('children.npy') or not os.path.exists('encodings.npy'):
        data.collect_images()
        children, encodings = model.train()

        np.save('children', children)
        np.save('encodings', encodings)

    children = np.load('children.npy')
    encodings = np.load('encodings.npy')
    return children, encodings


def uncompress_nparr(bytestring):
    return np.load(io.BytesIO(zlib.decompress(bytestring)))

@app.route('/recognize', methods=['POST'])
def recognize():
    rgb_small_frame = uncompress_nparr(request.data)
    return recog(rgb_small_frame)

def recog(rgb_small_frame):

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(encodings, face_encoding)
        name = "Unknown"

        # # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = children[best_match_index]

        face_names.append(name)

    return {'face_names': face_names, 'face_locations': face_locations}

if __name__ == '__main__':
    try:
        children, encodings = init()
        app.run()
    except Exception as e:
        print(e)
