import matplotlib.pyplot as plt
import base64
import os
from pymongo import MongoClient
from flask import Flask
from flask import request
import re
from train_v2 import train_for_a_person
import threading

app = Flask(__name__)

local = 'mongodb://127.0.0.1:27017'
server = 'mongodb+srv://usn012y2018:facepay1@facepay.y1chyja.mongodb.net/?retryWrites=true&w=majority'


client = MongoClient(server)
database = client.facepay

b64_imgs = 'b64_imgs'
user_name = 'user_name'
user_id = 'user_id'

@app.route("/train", methods=['POST'])
def index():
    req_body = request.json
    user_user_id = req_body[user_id]
    # if user_user_id not in os.listdir('./Faces'):
    #     os.mkdir(f'./Faces/{user_user_id}')
    imgs = []
    collection = database.images
    cursor = collection.find()
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
        img_file = open(f'./Faces/{user_user_id}/{req_body[user_id]}{str(index)}.{format}', 'wb')
        img_file.write(decoded_data)
        img_file.close()
    return_val = [None] * 1
    t = threading.Thread(target=train_for_a_person, args=(f'{user_user_id}', return_val))
    t.start()
    return f"Training Initiated for user {user_user_id}"



if __name__ == '__main__':
    try:
        app.run(port=9000)
    except:
        print('Could not start server')