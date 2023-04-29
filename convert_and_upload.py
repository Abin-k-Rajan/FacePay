from pymongo import MongoClient
import base64
import os


client = MongoClient('mongodb://127.0.0.1:27017')
database = client.facepay
collection = database.images

for folder in os.listdir('./Faces'):
    for file in os.listdir(f'./Faces/{folder}'):
        with open(f'./Faces/{folder}/{file}', 'rb') as im:
            b_64 = base64.b64encode(im.read())
        data = {"user_id": str(f'{folder}'), "img": b_64}
        collection.insert_one(data)
    print("inserted")
