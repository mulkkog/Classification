import os
import random
from glob import glob
import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from load_data import read_test_dataset, read_val_dataset
from inception_arch import inception_v4

# GPU 0만 보게 하려면:
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# Tensorflow seed
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)


random_seed = 1125
seed_everything(random_seed)

BATCH_SIZE = 32
EPOCHS = 200
NUM_CLASSES = 1000
LOG_DIR = 'logs/fit/' + datetime.datetime.now().strftime("%Y%m%d_%H:%M")
CP_DIR = 'checkpoint/' + datetime.datetime.now().strftime("%Y%m%d_%H:%M")

# Train data
TRAIN_IMG_DIR = '/home/jang/Disk_1TB/Dataset/ImageNet2012/train/'
TRAIN_TFR_DIR = '/home/jang/Disk_1TB/Dataset/ImageNet2012/TFRecords_split/train/'
TRAIN_NUM_EXAMPLES = len(glob(TRAIN_IMG_DIR + '/*/*'))

# Val data
VAL_IMG_DIR = '/home/jang/Disk_1TB/Dataset/ImageNet2012/val/'
VAL_TFR_DIR = '/home/jang/Disk_1TB/Dataset/ImageNet2012/TFRecords_split/val/'
VAL_NUM_EXAMPLES = len(glob(VAL_IMG_DIR + '/*/*'))

# Load the data
TRAINING_FILENAMES = tf.io.gfile.glob(f"{TRAIN_TFR_DIR}*.tfrecord")
VALID_FILENAMES = tf.io.gfile.glob(f"{VAL_TFR_DIR}*.tfrecord")

print("Train TFRecord Files:", len(TRAINING_FILENAMES))
print("Validation TFRecord Files:", len(VALID_FILENAMES))

# Read Data
train_dataset = read_test_dataset(TRAINING_FILENAMES, BATCH_SIZE)
val_dataset = read_val_dataset(VALID_FILENAMES, BATCH_SIZE)

image_batch, label_batch = next(iter(val_dataset))

classes = os.listdir(VAL_IMG_DIR)
classes.sort()


# Show Images and Labels
def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n] / 255.0)
        if label_batch[n]:
            label = label_batch[n]
            label = classes[label]
            plt.title(label)
        plt.axis("off")
    plt.show()


# show_batch(image_batch.numpy(), label_batch.numpy())

# TensorBoard callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(LOG_DIR)

# Checkpoint callback
checkpoint_path = CP_DIR + '/checkpoint' + '.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_best_only=True)

# Early stop callback
earlystop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    min_delta=0.00000001,
    patience=5)

'''
# Learning rate
initial_learning_rate = 0.045

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    # https://www.pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/
    decay_steps=TRAIN_NUM_EXAMPLES / BATCH_SIZE,
    decay_rate=0.94,
    staircase=True
)

# Train
optimizer = tf.keras.optimizers.RMSprop(epsilon=1.0, decay=0.9, learning_rate=lr_schedule)

model = inception_v4(NUM_CLASSES)
# model.load_weights('saved_models/googlenet_v1_imagenet.h5')

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])

# model.summary()

model.fit(train_dataset,
          epochs=EPOCHS,
          validation_data=val_dataset,
          validation_steps=int(VAL_NUM_EXAMPLES / BATCH_SIZE),
          steps_per_epoch=int(TRAIN_NUM_EXAMPLES / BATCH_SIZE),
          callbacks=[tensorboard_callback, cp_callback])
'''
# Learning rate
initial_lrate = 0.045


def lr_scheduler(epoch):
    lr = model.optimizer.lr.numpy()
    print(lr)
    if (epoch + 1) % 8 == 0:
        lr = lr * 0.94
    return lr


# Train
optimizer = tf.keras.optimizers.Adam()


lr_sc = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)

model = inception_v4(NUM_CLASSES)

model.compile(
    loss=['sparse_categorical_crossentropy'],
    optimizer=optimizer,
    metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(5)])

model.fit(train_dataset,
          epochs=EPOCHS,
          validation_data=val_dataset,
          validation_steps=int(VAL_NUM_EXAMPLES/BATCH_SIZE),
          steps_per_epoch=int(TRAIN_NUM_EXAMPLES/BATCH_SIZE),
          callbacks=[tensorboard_callback, cp_callback, lr_sc])

model.save('saved_models/inception_v4.h5')

print("Evaluate on val data")
results = model.evaluate(image_batch, label_batch)
print("val loss, val acc:", results)
