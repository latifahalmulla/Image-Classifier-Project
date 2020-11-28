import json
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import warnings 
warnings.filterwarnings('ignore')
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
import json
import glob 
from PIL import Image

from tensorflow.keras import layers

import argparse

parser = argparse.ArgumentParser(description='Image Classifier - Prediction Part')
parser.add_argument('--input', default='./test_images/hard-leaved_pocket_orchid.jpg', action="store", type = str, help='image path')
parser.add_argument('--model', default='./modell.h5', action="store", type = str, help='checkpoint file path/name')
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int, help='return top K most likely classes')
parser.add_argument('--category_names', dest="category_names", action="store", default='label_map.json', help='mapping the categories to real names')


arg_parser = parser.parse_args()

image_path = arg_parser.input
model_path = arg_parser.model
topk = arg_parser.top_k
category_names = arg_parser.category_names

def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    image=image.numpy()
    return image 

def predict(image_path, model, top_k):
    img = Image.open(image_path)
    test_img = np.asarray(img)
    trandform_img = process_image(test_img)
    redim_img = np.expand_dims(trandform_img, axis=0)
    prob_pred = model.predict(redim_img)
    prob_pred = prob_pred.tolist()
    
    probs, classes = tf.math.top_k(prob_pred, k=top_k)
    probs=probs.numpy().tolist()[0]
    classes=classes.numpy().tolist()[0]
    return probs,classes

if __name__== "__main__":

    print ("start Prediction ...")
    with open(category_names, 'r') as f:
        class_names = json.load(f)
    
    reloaded_model = tf.keras.models.load_model(model_path,custom_objects={'KerasLayer':hub.KerasLayer})
    probs, classes = predict(image_path, reloaded_model, topk)
    label_names = [class_names[str(int(idd)+1)] for idd in classes]
    print(probs)
    print(classes)
    print(label_names)
print ("End Prediction ")
