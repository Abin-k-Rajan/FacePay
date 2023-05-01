import matplotlib.pyplot as plt
import base64
import os
from pymongo import MongoClient
from flask import Flask
from flask import request, Response
import re
from detect import *
import threading
from flask_cors import CORS, cross_origin


app = Flask(__name__)
CORS(app)

local = 'mongodb://127.0.0.1:27017'
server = 'mongodb+srv://usn012y2018:facepay1@facepay.y1chyja.mongodb.net/?retryWrites=true&w=majority'


client = MongoClient(server)
database = client.facepay


@app.route("/update-encodings", methods=['POST'])
@cross_origin()
def update_encodings():
    print('Updating encodings')
    req_body = request.json
    timestamp = req_body['last-update']
    res = get_encoding_from_timestamp(timestamp=timestamp)
    if res['status'] == True:
        return Response(status=200, response=res['message'])
    return Response(status=404, response=res['message'])

@app.route("/update-encoding-for-user", methods=['POST'])
@cross_origin()
def update_encoding_for_user():
    req_body = request.json
    user_id = req_body['user_id']
    res = get_encoding_for_user(user_id=user_id)
    if res['status'] == True:
        return Response(status=200, response=res['message'])
    return Response(status=404, response=res['message'])




def capture_camera(face_detector , face_encoder):
    cap = cv2.VideoCapture(0)
    frames = []
    while cap.isOpened():
        ret,frame = cap.read()

        if not ret:
            print("CAM NOT OPEND") 
            break
        # coords = face_cascade.detectMultiScale(frame,scaleFactor=1.1, minNeighbors=5)
        res= detect(frame , face_detector , face_encoder)
        # frame = detect_opencv(frame, coords, face_encoder , encoding_dict)

        cv2.imshow('camera', res['img'])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



if __name__=='__main__':
    get_all_encodings_from_db()
    required_shape = (160,160)
    face_encoder = InceptionResNetV2()
    path_m = "facenet_keras_weights.h5"
    face_encoder.load_weights(path_m)
    encodings_path = './encodings'
    face_detector = mtcnn.MTCNN()
    load_pickle(encodings_path)
    


    t = threading.Thread(target=capture_camera,args=(face_detector, face_encoder))
    t.start()
    

    try:
        app.run(host='0.0.0.0', port=9090)
    except:
        print('Could not start server')
    