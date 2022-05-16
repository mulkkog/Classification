import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import math
from glob import glob
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential


# os optimize?
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'


# check if tensorflow see the GPU
print(tf.test.gpu_device_name())


# Securing Seeds
tf.random.set_seed(123)

# PATHS TO IMAGES
test_path = '/home/jang/Disk_1TB/Dataset/ImageNet2012/train/'
val_path = '/home/jang/Disk_1TB/Dataset/ImageNet2012/val/'

NUM_WORKERS = 2
IMG_SIZE = 224
BATCH_SIZE = 128
EPOCHS = 90


def build_alexnet(num_classes, img_size=224):
    # Create the model
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        # layer 1
        layers.Conv2D(filters=96, kernel_size=(11, 11),
                      strides=(4, 4), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Lambda(tf.nn.local_response_normalization),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # layer 2
        layers.Conv2D(256, kernel_size=(5, 5), activation='relu'),
        layers.Lambda(tf.nn.local_response_normalization),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # layer 3
        layers.Conv2D(384, kernel_size=(3, 3), activation='relu'),
        # layer 4
        layers.Conv2D(384, kernel_size=(3, 3), activation='relu'),
        # layer 5
        layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # layer 6
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        # layer 7
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        # layer 8
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def augmentation(image, label):
    image = tf.image.random_brightness(image, 1)
    image = tf.image.random_contrast(image, lower=0.0, upper=1.0)
    image = tf.image.random_flip_left_right(image)
    return image, label


def _parse_image_function(example):
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    features = tf.io.parse_single_example(example, image_feature_description)
    image = tf.image.decode_jpeg(features['image'], channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])

    label = tf.cast(features['label'], tf.int32)

    return image, label


def read_dataset(filename, batch_size):
    options = tf.data.Options()
    options.experimental_deterministic = False

    # dataset = tf.data.TFRecordDataset(filename)
    dataset = tf.data.Dataset.list_files(filename)
    dataset = dataset.with_options(options)
    dataset = dataset.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(_parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def use_tfrecords(path):
    train_dataset = read_dataset('/home/jang/Disk_1TB/Dataset/ImageNet2012/TFRecords/train/train2.tfrecords',
                                 BATCH_SIZE)
    val_dataset = read_dataset('/home/jang/Disk_1TB/Dataset/ImageNet2012/TFRecords/val/val3.tfrecords', BATCH_SIZE)

    num_classes = len(os.listdir(path))
    num_images = len(glob(path + '/*/*'))
    model = build_alexnet(num_classes, IMG_SIZE)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)])

    # Train the model

    # Tensorboard callback
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir)

    # Early stop callback
    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.0001,
        patience=5)

    # Checkpoint_old callback
    checkpoint_path = 'Checkpoint/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/checkpoint' + '.ckpt'
    # model.load_weights(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     monitor='val_loss',
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     save_freq='epoch')

    model.fit(train_dataset,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              steps_per_epoch=math.ceil(num_images / BATCH_SIZE),
              verbose=1,
              validation_data=val_dataset,
              validation_steps=1,
              validation_freq=1,
              callbacks=[tensorboard_callback,
                         earlystop_callback,
                         cp_callback])

    model.save('Models/alexnet_train4.h5')


use_tfrecords(test_path)
