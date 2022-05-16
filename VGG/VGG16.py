import tensorflow as tf


def vgg_16(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=(1, 1),
                               input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=(1, 1)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=(1, 1)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=(1, 1)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=(1, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=(1, 1)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', strides=(1, 1)),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', strides=(1, 1)),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', strides=(1, 1)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', strides=(1, 1)),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', strides=(1, 1)),
        tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', strides=(1, 1)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model
