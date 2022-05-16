# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

img_height = 224
img_width = 224
num_classes = 120


def classify(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    model = tf.keras.models.load_model('Models/stanford_dog_model.h5')
    prediction = model.predict(img_array)

    print(prediction)

    plt.imshow(img)
    plt.show()


classify("/home/jang/Disk_1TB/Dataset/Test/Bok1.jpg")

