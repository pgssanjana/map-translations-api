import uvicorn
import io
from fastapi import FastAPI
# from pollutants import Pollutants

import numpy as np
# import pickle
import pandas as pd
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import save_img
import tensorflow as tf
from numpy import expand_dims
import cv2
import base64
from PIL import Image
from type import GAN


app = FastAPI()


m1 =  load_model('/Users/sanjana/Downloads/sat2map.h5')
m2 =  load_model('/Users/sanjana/Downloads/map2sat.h5')

def load_image(filename, size=(256,256)):
    pixels = load_img(filename, target_size=size)
    pixels = img_to_array(pixels)
    pixels = (pixels - 127.5) / 127.5
    pixels = expand_dims(pixels, 0)
    return pixels

@app.get('/')
def home():
    return 'hello'

@app.post('/generate')
def generate(data : GAN):
    # print(data)
    # # image = flask.request.files['image'].read()
    data=data.dict()
    if(data['type']==0):
        model = m1
        # print('loaded model')
    else:
        model = m2
        # print('loaded model')
    # print(data)
    image = data['image']
    image = base64.b64decode(image)
    img_file = open('xyz.png', 'wb')
    img_file.write(image)
    img_file.close()
    # print(image)
    image = load_image('/Users/sanjana/Documents/sanjana/vscode/map-translations-fastapi/xyz.png')
    # print(image)
    preds=model.predict(image)
    output = tf.reshape(preds,[256,256,3])
    output = (output+1)/2.0
    # print(output)
    outputimg = 'xyz.png'
    save_img(outputimg,img_to_array(output))
    # import base64
    with open(outputimg, "rb") as img_file:
        my_string = base64.b64encode(img_file.read())
    # print(my_string)
    response ={
        "image" : my_string
    }
    return response


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000)