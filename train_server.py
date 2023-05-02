import matplotlib.pyplot as plt
import base64
import os
from pymongo import MongoClient
from flask import Flask
from flask import request
import re
from train_v2 import train_for_a_person, set_proxy_address
import threading
from flask_cors import CORS, cross_origin
import sys
import requests



app = Flask(__name__)
CORS(app)

local = 'mongodb://127.0.0.1:27017'
server = 'mongodb+srv://usn012y2018:facepay1@facepay.y1chyja.mongodb.net/?retryWrites=true&w=majority'


client = MongoClient(server)
database = client.facepay

b64_imgs = 'b64_imgs'
user_name = 'user_name'
user_id = 'user_id'


def start_training(user_user_id):
    imgs = []
    collection = database.images
    cursor = collection.find({"user_id": user_user_id})
    for val in cursor:
        imgs.append(val['img'])
    for index, val in enumerate(imgs):
        val = str(val)
        res = re.search(",", str(val))
        decoded_data = base64.b64decode((val[res.start() + 1:]))
        format = 'jpg'
        if re.search('jpeg', val[0:100]):
            format = 'jpg'
        elif re.search('png', val[0:100]):
            format = 'png'
        img_file = open(f'./Faces/{user_user_id}/{user_user_id}{str(index)}.{format}', 'wb')
        img_file.write(decoded_data)
        img_file.close()
    return_val = [None] * 1
    t = threading.Thread(target=train_for_a_person, args=(f'{user_user_id}', return_val))
    t.start()


@app.route("/train", methods=['POST'])
@cross_origin()
def index():
    user_user_id = request.args.get('user_id')
    if user_user_id not in os.listdir('./Faces'):
        os.mkdir(f'./Faces/{user_user_id}')
    t = threading.Thread(target=start_training, args=[user_user_id])
    t.start()
    return f"Training Initiated for user {user_user_id}"



if __name__ == '__main__':
    PORT = 9000
    NODE = 'train-server'
    if len(sys.argv) < 2:
        print('\n\n\n Error in Usage: python train_server.py <Proxy Address>\n\n\n')
    else:
        proxy_address = sys.argv[1]
        response = requests.post(f'http://{proxy_address}:5000/init?node={NODE}&port={PORT}')
        print(response.content)
        proxy_address = sys.argv[1]
        set_proxy_address(proxy_address)
        if 'encodings' not in os.listdir():
            os.makedirs('encodings')
        if 'Faces' not in os.listdir():
            os.makedirs('Faces')
        try:
            app.run(host='0.0.0.0', port=PORT, debug=True)
        except:
            print('Could not start server')