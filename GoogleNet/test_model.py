# TensorFlow and tf.keras
import os

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = 224
IMG_DIR = '/home/jang/Disk_1TB/Dataset/ImageNet2012/train/'


classes = os.listdir(IMG_DIR)
classes.sort()


def classify(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    model = tf.keras.models.load_model('saved_models/m3.h5')
    prediction = model.predict(img_array)

    '''
    prediction[0]
    label = np.argmax(prediction[0])
    name = classes[label]
    '''

    top_1 = '1. ' + classes[np.argmax(prediction[0])]
    top_2 = '\n2. ' + classes[np.argmax(prediction[1])]
    top_3 = '\n3. ' + classes[np.argmax(prediction[2])]

    name = top_1 + top_2 + top_3

    print(name)
    plt.figure(figsize=(10, 8))
    plt.title(name)
    plt.imshow(img)
    plt.savefig('Bok1.png')
    plt.show()


classify("/home/jang/Disk_1TB/Dataset/Test/Bok1.jpg")

