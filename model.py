import os
import sys

import PIL
import face_recognition
import numpy as np

import constants


class Model:

    def __init__(self, _children, _encodings):
        self.children = _children
        self.encodings = _encodings

    def get_children(self):
        return self.children

    def get_encodings(self):
        return self.encodings

    def train(self):
        for root, _, files in os.walk(constants.DATASET_LOCATION):
            if len(files) > 0:
                print(root, files)
                child_id = os.path.basename(root)
                file = files[0]
                self.train_from_single_img(child_id, root + '/' + file)
        print('Successfully trained model using images')
        return self.children, self.encodings

    def train_from_single_img(self, child_id, file):
        try:
            child_img = face_recognition.load_image_file(file)
            child_face_encodings = face_recognition.face_encodings(child_img)
            if len(child_face_encodings) > 0:
                self.encodings.append(child_face_encodings[0])
                self.children.append(child_id)
        except PIL.UnidentifiedImageError as e:
            print(e)


if __name__ == '__main__':
    try:
        children_exist = os.path.exists('./children.npy')
        encodings_exist = os.path.exists('./encodings.npy')

        if not children_exist or not encodings_exist:
            print('Error pre trained model does not exists')
            exit()

        children = np.load('children.npy', allow_pickle=True).tolist()
        encodings = np.load('encodings.npy', allow_pickle=True).tolist()

        model = Model(children, encodings)
        model.train_from_single_img(sys.argv[1], constants.CHILD_DATASET_LOCATION_FORMAT.format(sys.argv[2]))
        np.save('children', model.get_children())
        np.save('encodings', model.get_encodings())

    except Exception as e:
        print(e)
