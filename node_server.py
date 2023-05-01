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
import sys
import requests


app = Flask(__name__)
CORS(app)

NODE = ''
proxy_address = '127.0.0.1'
local = 'mongodb://127.0.0.1:27017'
server = 'mongodb+srv://usn012y2018:facepay1@facepay.y1chyja.mongodb.net/?retryWrites=true&w=majority'


name = 'unknown'
balance = 'unknown'
status = 'unknown'


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
    user_id = request.args.get('user_id')
    res = get_encoding_for_user(user_id=user_id)
    if res['status'] == True:
        return Response(status=200, response=res['message'])
    return Response(status=404, response=res['message'])


def user_entry_or_exit(userID):
    global NODE
    global name
    global status
    global balance
    response = requests.post(f'http://{proxy_address}:5000/backend-server/validate?userID={userID}&stationID={NODE}')
    sample_json = '\{"name" : "John", "status" : "Active", "balance": 78\}'
    name = "John"
    status = 'Active'
    balance = 78
    



def capture_camera(face_detector , face_encoder):
    global name
    global status
    global balance
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
        if res['id'] != name and res['id'] != 'unknown':
            t = threading.Thread(target = user_entry_or_exit, args = [res['id']])
            t.start()
            name = res['id']
        img = res['img']

        cv2.putText(img, f'Name: {name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f'Status: {status}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f'Balance: {balance}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(img, balance, 12, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.imshow('FACEPAY', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def set_station_id(station_id):
    global NODE
    NODE = station_id

def set_proxy_address(proxy):
    global proxy_address
    proxy_address = proxy



if __name__=='__main__':
    PORT = 9090
    if len(sys.argv) < 3:
        print('Error in Usage: python node_server.py station_id proxy_ip')
    else:
        NODE = sys.argv[1]
        set_station_id(NODE)
        proxy_address = sys.argv[2]
        set_proxy_address(proxy_address)
        response = requests.post(f'http://{proxy_address}:5000/init?node={NODE}&port={PORT}')
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
            app.run(host='0.0.0.0', port=PORT)
        except:
            print('Could not start server')
    