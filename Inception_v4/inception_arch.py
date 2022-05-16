import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Input, concatenate, AveragePooling2D, Flatten

kernel_init = keras.initializers.he_normal()


# Stem module
def stem(input_layer):
    conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu',
                      kernel_initializer=kernel_init)(input_layer)
    conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                      kernel_initializer=kernel_init)(conv_3x3)
    conv_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                      kernel_initializer=kernel_init)(conv_3x3)

    # branch 1
    max_pool_3x3_b1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv_3x3)

    # branch 2
    conv_3x3_b2 = Conv2D(filters=96, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu',
                         kernel_initializer=kernel_init)(conv_3x3)

    filter_concat = concatenate([max_pool_3x3_b1, conv_3x3_b2])

    # branch 1
    conv_1x1_b1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(filter_concat)
    conv_3x3_b1 = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                         kernel_initializer=kernel_init)(conv_1x1_b1)

    # branch 2
    conv_1x1_b2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(filter_concat)
    conv_7x1_b2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(conv_1x1_b2)
    conv_1x7_b2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(conv_7x1_b2)
    conv_3x3_b2 = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                         kernel_initializer=kernel_init)(conv_1x7_b2)

    filter_concat = concatenate([conv_3x3_b1, conv_3x3_b2])

    # branch 1
    conv_3x3_b1 = Conv2D(filters=192, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu',
                         kernel_initializer=kernel_init)(filter_concat)
    # branch 2
    max_pool_b2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(filter_concat)

    filter_concat = concatenate([conv_3x3_b1, max_pool_b2])
    # print(filter_concat)

    return filter_concat


# Inception-A
def inception_a(filter_concat):
    # branch 1
    avg_pool_b1 = AveragePooling2D(pool_size=(1, 1))(filter_concat)
    conv_1x1_b1 = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(avg_pool_b1)

    # branch 2
    conv_1x1_b2 = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(filter_concat)

    # branch 3
    conv_1x1_b3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(filter_concat)
    conv_3x3_b3 = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(conv_1x1_b3)

    # branch 4
    conv_1x1_b4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(filter_concat)
    conv_3x3_b4 = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(conv_1x1_b4)
    conv_3x3_b4 = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(conv_3x3_b4)

    # concat
    filter_concat = concatenate([conv_1x1_b1, conv_1x1_b2, conv_3x3_b3, conv_3x3_b4])

    return filter_concat


# Reduction-A
def reduction_a(filter_concat):
    # branch 1
    max_pool_b1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(filter_concat)

    # branch 2
    conv_3x3_b2 = Conv2D(filters=384, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu',
                         kernel_initializer=kernel_init)(filter_concat)

    # branch 3
    conv_1x1_b3 = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(filter_concat)
    conv_3x3_b3 = Conv2D(filters=224, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(conv_1x1_b3)
    conv_3x3_b3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu',
                         kernel_initializer=kernel_init)(conv_3x3_b3)

    # concat
    filter_concat = concatenate([max_pool_b1, conv_3x3_b2, conv_3x3_b3])

    return filter_concat


# Inception-B
def inception_b(filter_concat):
    # branch 1
    avg_pool_b1 = AveragePooling2D(pool_size=(1, 1))(filter_concat)
    conv_1x1_b1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(avg_pool_b1)

    # branch 2
    conv_1x1_b2 = Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(filter_concat)

    # branch 3
    conv_1x1_b3 = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(filter_concat)
    conv_1x7_b3 = Conv2D(filters=224, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(conv_1x1_b3)
    conv_7x1_b3 = Conv2D(filters=256, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(conv_1x7_b3)

    # branch 4
    conv_1x1_b4 = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(filter_concat)
    conv_1x7_b4 = Conv2D(filters=192, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(conv_1x1_b4)
    conv_7x1_b4 = Conv2D(filters=224, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(conv_1x7_b4)
    conv_1x7_b4 = Conv2D(filters=224, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(conv_7x1_b4)
    conv_7x1_b4 = Conv2D(filters=256, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(conv_1x7_b4)

    # concat
    filter_concat = concatenate([conv_1x1_b1, conv_1x1_b2, conv_7x1_b3, conv_7x1_b4])

    return filter_concat


# Reduction-B
def reduction_b(filter_concat):
    # branch 1
    max_pool_b1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(filter_concat)

    # branch 2
    conv_1x1_b2 = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(filter_concat)
    conv_3x3_b2 = Conv2D(filters=192, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu',
                         kernel_initializer=kernel_init)(conv_1x1_b2)

    # branch 3
    conv_1x1_b4 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(filter_concat)
    conv_1x7_b4 = Conv2D(filters=256, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(conv_1x1_b4)
    conv_7x1_b4 = Conv2D(filters=320, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(conv_1x7_b4)
    conv_3x3_b4 = Conv2D(filters=320, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu',
                         kernel_initializer=kernel_init)(conv_7x1_b4)

    # concat
    filter_concat = concatenate([max_pool_b1, conv_3x3_b2, conv_3x3_b4])

    return filter_concat


# Inception-C
def inception_c(filter_concat):
    # branch 1
    avg_pool_b1 = AveragePooling2D(pool_size=(1, 1))(filter_concat)
    conv_1x1_b1 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(avg_pool_b1)

    # branch 2
    conv_1x1_b2 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(filter_concat)

    # branch 3
    conv_1x1_b3 = Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(filter_concat)
    conv_1x3_b3_1 = Conv2D(filters=256, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu',
                           kernel_initializer=kernel_init)(conv_1x1_b3)
    conv_3x1_b3_2 = Conv2D(filters=256, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu',
                           kernel_initializer=kernel_init)(conv_1x1_b3)
    concat_b3 = concatenate([conv_1x3_b3_1, conv_3x1_b3_2])

    # branch 4
    conv_1x1_b4 = Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(filter_concat)
    conv_1x3_b4 = Conv2D(filters=448, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(conv_1x1_b4)
    conv_3x1_b4 = Conv2D(filters=512, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=kernel_init)(conv_1x3_b4)
    conv_3x1_b4_1 = Conv2D(filters=256, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu',
                           kernel_initializer=kernel_init)(conv_3x1_b4)
    conv_1x3_b4_2 = Conv2D(filters=256, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu',
                           kernel_initializer=kernel_init)(conv_3x1_b4)
    concat_b4 = concatenate([conv_3x1_b4_1, conv_1x3_b4_2])

    # concat
    filter_concat = concatenate([conv_1x1_b1, conv_1x1_b2, concat_b3, concat_b4])

    return filter_concat


# Inception_v4
def inception_v4(num_classes):
    input_layer = Input(shape=(299, 299, 3))

    # Stem
    x = stem(input_layer)

    # Inception-A
    x = inception_a(x)
    x = inception_a(x)
    x = inception_a(x)
    x = inception_a(x)

    # Reduction-A
    x = reduction_a(x)

    # Inception-B
    x = inception_b(x)
    x = inception_b(x)
    x = inception_b(x)
    x = inception_b(x)
    x = inception_b(x)
    x = inception_b(x)
    x = inception_b(x)

    # Reduction-B
    x = reduction_b(x)

    # Inception-C
    x = inception_c(x)
    x = inception_c(x)
    x = inception_c(x)
    print(x)

    # Average-Pooling
    x = AveragePooling2D()(x)

    # Dropout
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)

    # Softmax
    x = Dense(num_classes, activation='softmax')(x)

    return Model(input_layer, x, name='inception_v4')
