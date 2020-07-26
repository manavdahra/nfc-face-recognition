import io
import os
import zlib

import face_recognition
import numpy as np
from flask import Flask
from flask import request

import data
import model

app = Flask(__name__)
server = {}

def uncompress_nparr(bytestring):
    return np.load(io.BytesIO(zlib.decompress(bytestring)))


class Server:

    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []

        children_exist = os.path.exists('./children.npy')
        encodings_exist = os.path.exists('./encodings.npy')

        if not children_exist or not encodings_exist:
            data.collect_images()
            _model = model.Model([], [])
            _model.train()
            _children = _model.get_children()
            _encodings = _model.get_encodings()
        else:
            _children = np.load('children.npy')
            _encodings = np.load('encodings.npy')

        np.save('children', _children)
        np.save('encodings', _encodings)

        self.children = _children
        self.encodings = _encodings

    def recognize(self, rgb_small_frame):
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.children[best_match_index]

            face_names.append(name)

        return {'face_names': face_names, 'face_locations': face_locations}


@app.route('/recognize', methods=['POST'])
def recognize():
    rgb_small_frame = uncompress_nparr(request.data)
    return server.recognize(rgb_small_frame)


if __name__ == '__main__':
    try:
        server = Server()
        app.run()
    except Exception as e:
        print(e)
