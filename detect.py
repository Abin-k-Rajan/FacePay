import cv2 
import numpy as np
import mtcnn
from architecture import *
from train_v2 import normalize,l2_normalizer
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model
import pickle
import mediapipe as mp
from facenet_pytorch import MTCNN
from pymongo import MongoClient
import json
import os


import numpy as np
import torch
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

import pickle


client = MongoClient('mongodb://127.0.0.1:27017')
database = client.facepay
path = 'encodings/encodings.pkl'

encodings_path = './encodings'
encodings = []


def get_all_encodings_from_db():
    collection = database.encoding
    encodings = collection.find()
    _encodings = {}
    for encode in encodings:
        _encodings[encode['user_id']] = encode['encoding']
    for user_id, encode in _encodings.items():
        with open(f'{encodings_path}/{user_id}.pkl', 'wb') as file:
            enc = {}
            enc[user_id] = encode
            pickle.dump(enc, file)
        

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from blazeface import BlazeFace



front_net = BlazeFace().to(gpu)
front_net.load_weights("blazeface.pth")
front_net.load_anchors("anchors.npy")
back_net = BlazeFace(back_model=True).to(gpu)
back_net.load_weights("blazefaceback.pth")
back_net.load_anchors("anchorsback.npy")


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

confidence_t=0.99
recognition_t=0.3
required_size = (160,160)

def get_face(img, box):
    x1, y1, x2, y2 = box[0]
    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)
    # x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode


def load_pickle(path):
    encoding_dict = {}
    for enc in os.listdir(f'./{encodings_path}'):
        if enc == '.git':
            continue
        with open(f'./{encodings_path}/{enc}', 'rb') as f:
            encodings.append(pickle.load(f))
    for enc in encodings:
        for name, encode in enc.items():
            encoding_dict[name] = encode
    return encoding_dict



def detect(img ,detector,encoder,encoding_dict):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_face_detection = mp.solutions.face_detection
    mtcnn = MTCNN()
    face, prob = mtcnn.detect(img_rgb)
    # results = detector.detect_faces(img_rgb)
    # for res in results:
    #     if res['confidence'] < confidence_t:
    #         continue
    if face is None:
        pass
    else:
        face, pt_1, pt_2 = get_face(img_rgb, face)
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
    return img 


def detect_opencv(img, coords, encoder, encoding_dict):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if type(coords) == tuple:
        return img
    x, y, w, h = coords[0]
    pt_1 = (x, y)
    pt_2 = (x+w, y+h)
    face = img_rgb[x:x+w, y:y+h]
    encode = get_encode(encoder, face, required_size)
    encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
    name = 'unknown'

    distance = float("inf")
    for db_name, db_encode in encoding_dict.items():
        dist = cosine(db_encode, encode)
        if dist < recognition_t and dist < distance:
            name = db_name
            distance = dist

    if name == 'unknown':
        cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
        cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    else:
        cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
        cv2.putText(img, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 200, 200), 2)
    return img 




if __name__ == "__main__":
    get_all_encodings_from_db()
    required_shape = (160,160)
    face_encoder = InceptionResNetV2()
    path_m = "facenet_keras_weights.h5"
    face_encoder.load_weights(path_m)
    encodings_path = './encodings'
    face_detector = mtcnn.MTCNN()
    encoding_dict = load_pickle(encodings_path)
    
    cap = cv2.VideoCapture(0)
    frames = []

    while cap.isOpened():
        ret,frame = cap.read()

        if not ret:
            print("CAM NOT OPEND") 
            break
        # coords = face_cascade.detectMultiScale(frame,scaleFactor=1.1, minNeighbors=5)
        frame= detect(frame , face_detector , face_encoder , encoding_dict)
        # frame = detect_opencv(frame, coords, face_encoder , encoding_dict)

        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # get_all_encodings_from_db()
    


