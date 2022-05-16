import functools

import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


# Get Data
def _parse_image_function(example, input_shape, augment=False):
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    features = tf.io.parse_single_example(example, image_feature_description)
    image = tf.image.decode_jpeg(features['image'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    if augment:
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)

        # Random B/S change
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.clip_by_value(image, 0.0, 1.0)  # keeping pixel values in check

        # Random resize and random crop back to expected size:
        random_scale_factor = tf.random.uniform([1], minval=1., maxval=1.4, dtype=tf.float32)
        scaled_height = tf.cast(tf.cast(input_shape[0], tf.float32) * random_scale_factor,
                                tf.int32)
        scaled_width = tf.cast(tf.cast(input_shape[1], tf.float32) * random_scale_factor,
                               tf.int32)
        scaled_shape = tf.squeeze(tf.stack([scaled_height, scaled_width]))
        image = tf.image.resize(image, scaled_shape)
        image = tf.image.random_crop(image, input_shape)
    else:
        image = tf.image.resize(image, input_shape[:2])

    label = tf.cast(features['label'], tf.int32)

    return image, label


def read_dataset(filename, batch_size, input_shape, is_train=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.with_options(ignore_order)

    if is_train:
        _parse_image_function_for_train = functools.partial(_parse_image_function,
                                                            input_shape=input_shape,
                                                            augment=True)
        dataset = dataset.map(_parse_image_function_for_train, num_parallel_calls=AUTOTUNE)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    else:
        _parse_image_function_for_train = functools.partial(_parse_image_function,
                                                            input_shape=input_shape,
                                                            augment=False)
        dataset = dataset.map(_parse_image_function_for_train, num_parallel_calls=AUTOTUNE)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset


