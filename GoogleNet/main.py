import os
from glob import glob
import keras
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import numpy as np
import tensorflow_datasets as tfds
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Input, concatenate, AveragePooling2D, Flatten
from tensorflow.python.keras.callbacks import LearningRateScheduler

tf.random.set_seed(1234)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

BATCH_SIZE = 32
EPOCHS = 100
NUM_CLASSES = 1000
LOG_DIR = 'logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
CP_DIR = 'checkpoint/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
IMG_DIR = '/home/jang/Disk_1TB/Dataset/ImageNet2012/train/'
TFR_DIR = '/home/jang/Disk_1TB/Dataset/ImageNet2012/TFRecords_split/train/'
AUTOTUNE = tf.data.experimental.AUTOTUNE
NUM_EXAMPLES = len(glob(IMG_DIR + '/*/*'))

kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model


# Inception module
def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5,
                     filters_pool_proj, name=None):
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(x)

    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
                       bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

    return output


# GoogLeNet
def googlenet(num_classes, with_classifier=False):
    input_layer = Input(shape=(224, 224, 3))

    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2',
               kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
    x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

    x = inception_module(x,
                         filters_1x1=64,
                         filters_3x3_reduce=96,
                         filters_3x3=128,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32,
                         name='inception_3a')

    x = inception_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=192,
                         filters_5x5_reduce=32,
                         filters_5x5=96,
                         filters_pool_proj=64,
                         name='inception_3b')

    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

    x = inception_module(x,
                         filters_1x1=192,
                         filters_3x3_reduce=96,
                         filters_3x3=208,
                         filters_5x5_reduce=16,
                         filters_5x5=48,
                         filters_pool_proj=64,
                         name='inception_4a')

    classifier_1 = AveragePooling2D((5, 5), strides=3)(x)
    classifier_1 = Conv2D(128, (1, 1), padding='same', activation='relu')(classifier_1)
    classifier_1 = Flatten()(classifier_1)
    classifier_1 = Dense(1024, activation='relu')(classifier_1)
    classifier_1 = Dropout(0.7)(classifier_1)
    classifier_1 = Dense(num_classes, activation='softmax', name='auxilliary_output_1')(classifier_1)

    x = inception_module(x,
                         filters_1x1=160,
                         filters_3x3_reduce=112,
                         filters_3x3=224,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4b')

    x = inception_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=256,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4c')

    x = inception_module(x,
                         filters_1x1=112,
                         filters_3x3_reduce=144,
                         filters_3x3=288,
                         filters_5x5_reduce=32,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4d')

    classifier_2 = AveragePooling2D((5, 5), strides=3)(x)
    classifier_2 = Conv2D(128, (1, 1), padding='same', activation='relu')(classifier_2)
    classifier_2 = Flatten()(classifier_2)
    classifier_2 = Dense(1024, activation='relu')(classifier_2)
    classifier_2 = Dropout(0.7)(classifier_2)
    classifier_2 = Dense(num_classes, activation='softmax', name='auxilliary_output_2')(classifier_2)

    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_4e')

    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5a')

    x = inception_module(x,
                         filters_1x1=384,
                         filters_3x3_reduce=192,
                         filters_3x3=384,
                         filters_5x5_reduce=48,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5b')

    x = AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid', name='avg_pool_5_3x3/1')(x)

    x = Dropout(0.4)(x)
    x = Dense(num_classes, activation='relu', name='linear')(x)
    x = Dense(num_classes, activation='softmax', name='output')(x)

    if with_classifier:
        return Model(input_layer, [x, classifier_1, classifier_2], name='googlenet_complete_architecture')

    return Model(input_layer, [x], name='googlenet')


# Preprocess
def preprocess(dataset):
    image = tf.image.resize(dataset["image"], size=(224, 224))
    return image, dataset['label']


# Get Data
def _parse_image_function(example):
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    features = tf.io.parse_single_example(example, image_feature_description)
    image = tf.image.decode_jpeg(features['image'], channels=3)
    image = tf.image.resize(image, [224, 224])

    label = tf.cast(features['label'], tf.int32)

    return image, label


def read_dataset(filename):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(_parse_image_function, num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(1028)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


# Load the data
FILENAMES = tf.io.gfile.glob(f"{TFR_DIR}*.tfrecord")
split_ind = int(0.9 * len(FILENAMES))
TRAINING_FILENAMES, VALID_FILENAMES = FILENAMES[:split_ind], FILENAMES[split_ind:]

TEST_FILENAMES = tf.io.gfile.glob("/home/jang/Disk_1TB/Dataset/Stanford_Dog/TFRecord/Split/test.tfrecord")
print("Train TFRecord Files:", len(TRAINING_FILENAMES))
print("Validation TFRecord Files:", len(VALID_FILENAMES))
print("Test TFRecord Files:", len(TEST_FILENAMES))

# Read Data
train_dataset = read_dataset(VALID_FILENAMES)
val_dataset = read_dataset(TRAINING_FILENAMES)
'''
train_dataset = read_dataset(TRAINING_FILENAMES)
valid_dataset = read_dataset(VALID_FILENAMES)
test_dataset = read_dataset(TEST_FILENAMES)
'''

image_batch, label_batch = next(iter(train_dataset))

classes = os.listdir(IMG_DIR)
classes.sort()


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n] / 255.0)
        if label_batch[n]:
            label = label_batch[n]
            label = classes[label]
            # name = label.split('-')[1]
            plt.title(label)
        plt.axis("off")
    plt.show()


show_batch(image_batch.numpy(), label_batch.numpy())

# TensorBoard callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(LOG_DIR)

# Checkpoint callback
checkpoint_path = CP_DIR + '/checkpoint' + '.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_best_only=True)

# Learning rate
initial_lrate = 0.001


def decay(epoch, steps=100):
    initial_lrate = 0.001
    drop = 0.96
    epochs_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


# Train
num_images = len(glob(IMG_DIR + '/*/*'))
sgd = tf.keras.optimizers.SGD(momentum=0.9, learning_rate=initial_lrate)

lr_sc = LearningRateScheduler(decay, verbose=1)

model = googlenet(NUM_CLASSES, True)

model.compile(
    loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'],
    loss_weights=[1, 0.3, 0.3],
    optimizer=sgd,
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
model.fit(train_dataset,
          epochs=EPOCHS,
          # steps_per_epoch=int(NUM_EXAMPLES / BATCH_SIZE),
          validation_data=val_dataset,
          callbacks=[tensorboard_callback, cp_callback, lr_sc])

model.save('saved_models/m3.h5')
