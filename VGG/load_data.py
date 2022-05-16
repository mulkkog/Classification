import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


# Get Data
def _parse_image_function(example):
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    features = tf.io.parse_single_example(example, image_feature_description)
    image = tf.image.decode_jpeg(features['image'], channels=3)


    # mean = [0.485, 0.456, 0.406]
    label = tf.cast(features['label'], tf.int32)

    return image, label


def read_test_dataset(filename, batch_size):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(_parse_image_function, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(lambda image, label: (tf.image.resize(image, size=[286, 286]), label),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)  # resize
    dataset = dataset.map(lambda image, label: (tf.image.random_flip_left_right(image), label),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)  # flip
    dataset = dataset.map(lambda image, label: (tf.image.random_crop(image, size=[224, 224, 3]), label),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)  # crop
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=10)
    return dataset


def read_val_dataset(filename, batch_size):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(_parse_image_function, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(lambda image, label: (tf.image.resize(image, size=[224, 224]), label),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)  # resize
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset
