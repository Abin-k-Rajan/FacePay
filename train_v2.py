from architecture import * 
import os 
import cv2
import mtcnn
import pickle 
import numpy as np 
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import load_model


import numpy as np
import torch
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from pymongo import MongoClient
import bson
import time
import datetime

client = MongoClient('mongodb://127.0.0.1:27017')
database = client.facepay


print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from blazeface import BlazeFace
import threading



front_net = BlazeFace().to(gpu)
front_net.load_weights("blazeface.pth")
front_net.load_anchors("anchors.npy")
back_net = BlazeFace(back_model=True).to(gpu)
back_net.load_weights("blazefaceback.pth")
back_net.load_anchors("anchorsback.npy")

######pathsandvairables#########
face_data = 'Faces/'
required_shape = (160,160)
face_encoder = InceptionResNetV2()
path = "facenet_keras_weights.h5"
face_encoder.load_weights(path)
face_detector = mtcnn.MTCNN()
encodes = []
encoding_dict = dict()
l2_normalizer = Normalizer('l2')
###############################


def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def load_encoding_into_database(encoding):
    collection = database.encoding
    for user_id, encode in encoding.items():
        data = {"user_id": user_id, "encoding": list(encode.tolist()), "last-update": datetime.datetime.now()}
        collection.insert_one(data)
    print(f'Encoding updated for user {user_id}')
    # Create a post request to Backend notifying update

def get_all_encodings_from_db():
    collection = database.encoding
    data = collection.find()
    for val in data:
        encoding = val['encoding']
        encoding = eval(encoding)
        print(type(encoding))
    


def train_for_a_person(face_id, return_val):
    face_id = face_id
    directory = f'./Faces/{face_id}'
    if not os.listdir(directory):
        return {"status": False, "reason": 'Directory / User Not Found'}
    for image_name in os.listdir(directory):
        image_path = os.path.join(directory,image_name)

        img_BGR = cv2.imread(image_path)
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

        face_img_resized = cv2.resize(img_RGB, (256,256))

        x = back_net.predict_on_image(face_img_resized)

        xmin = ymin = xmax = ymax = 0
    
        for i in range(x.shape[0]):
            ymin = math.floor(x[i, 0] * img_RGB.shape[0])
            xmin = math.floor(x[i, 1] * img_RGB.shape[1])
            ymax = math.floor(x[i, 2] * img_RGB.shape[0])
            xmax = math.floor(x[i, 3] * img_RGB.shape[1])

        img_RGB = img_RGB[ymin:ymax , xmin:xmax]

        x = face_detector.detect_faces(img_RGB)
        if x == []:
            continue
        x1, y1, width, height = x[0]['box']
        x1, y1 = abs(x1) , abs(y1)
        x2, y2 = x1+width , y1+height
        face = img_RGB[y1:y2 , x1:x2]
        
        face = normalize(face)
        face = cv2.resize(face, required_shape)
        face_d = np.expand_dims(face, axis=0)
        encode = face_encoder.predict(face_d)[0]
        encodes.append(encode)

    if encodes:
        encode = np.sum(encodes, axis=0 )
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        encoding_dict[face_id] = encode
    path = f'encodings/encodings{face_id}.pkl'
    # with open(path, 'wb') as file:
    #     pickle.dump(encoding_dict, file)
    print(f'Saved encoding of user {face_id}')
    th = threading.Thread(target=load_encoding_into_database, args=[encoding_dict])
    th.start()
    return encode



def train_all():
    for face_names in os.listdir(face_data):
        person_dir = os.path.join(face_data,face_names)

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir,image_name)

            img_BGR = cv2.imread(image_path)
            img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

            face_img_resized = cv2.resize(img_RGB, (256,256))

            x = back_net.predict_on_image(face_img_resized)

            xmin = ymin = xmax = ymax = 0
        
            for i in range(x.shape[0]):
                ymin = math.floor(x[i, 0] * img_RGB.shape[0])
                xmin = math.floor(x[i, 1] * img_RGB.shape[1])
                ymax = math.floor(x[i, 2] * img_RGB.shape[0])
                xmax = math.floor(x[i, 3] * img_RGB.shape[1])

            img_RGB = img_RGB[ymin:ymax , xmin:xmax]

            x = face_detector.detect_faces(img_RGB)
            if x == []:
                continue
            x1, y1, width, height = x[0]['box']
            x1, y1 = abs(x1) , abs(y1)
            x2, y2 = x1+width , y1+height
            face = img_RGB[y1:y2 , x1:x2]
            
            face = normalize(face)
            face = cv2.resize(face, required_shape)
            face_d = np.expand_dims(face, axis=0)
            encode = face_encoder.predict(face_d)[0]
            encodes.append(encode)

        if encodes:
            encode = np.sum(encodes, axis=0 )
            encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
            encoding_dict[face_names] = encode
        
    path = 'encodings/encodings.pkl'
    with open(path, 'wb') as file:
        pickle.dump(encoding_dict, file)





if __name__ == '__main__':
    get_all_encodings_from_db()