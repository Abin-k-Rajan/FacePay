from pymongo import MongoClient
import base64
import os
import re
server = 'mongodb+srv://usn012y2018:facepay1@facepay.y1chyja.mongodb.net/?retryWrites=true&w=majority'

client = MongoClient(server)
database = client.facepay
collection = database.images

for folder in os.listdir('./Faces'):
    for file in os.listdir(f'./Faces/{folder}'):
        with open(f'./Faces/{folder}/{file}', 'rb') as im:
            b_64 = base64.b64encode(im.read())
            b_64 = b_64.decode()
        if re.search(".png", file):
            b_64 = 'data:image/png;base64,' + b_64
        else:
            b_64 = 'data:image/jpeg;base64,'+b_64
        data = {"user_id": str(f'{folder}'), "img": str(b_64)}
        collection.insert_one(data)
    print("inserted")
