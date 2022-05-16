import os
from glob import glob
import datetime
import collections
import tensorflow as tf
import matplotlib.pyplot as plt
from load_data import read_dataset
from ResNet import ResNet18

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.random.set_seed(1234)

# Hyper Parameters
INPUT_SHAPE = [224, 224, 3]
BATCH_SIZE = 128
EPOCHS = 10
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
train_dataset = read_dataset(TRAINING_FILENAMES, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE, is_train=True)
val_dataset = read_dataset(VALID_FILENAMES, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE, is_train=False)

'''
image_batch, label_batch = next(iter(train_dataset))

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


show_batch(image_batch.numpy(), label_batch.numpy())
'''

ResNet18 = ResNet18(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
ResNet18.summary()

optimizer = tf.keras.optimizers.Adam()  # tf.keras.optimizers.SGD(momentum=0.9, nesterov=True)

accuracy_metric = tf.metrics.SparseCategoricalAccuracy(name='acc')
top5_accuracy_metric = tf.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_acc')
ResNet18.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                 metrics=[accuracy_metric, top5_accuracy_metric])

# Setting some variables to format the logs:
log_begin_red, log_begin_blue, log_begin_green = '\033[91m', '\033[94m', '\033[92m'
log_begin_bold, log_begin_underline = '\033[1m', '\033[4m'
log_end_format = '\033[0m'


class SimpleLogCallback(tf.keras.callbacks.Callback):
    """ Keras callback for simple, denser console logs."""

    def __init__(self, metrics_dict, num_epochs='?', log_frequency=1,
                 metric_string_template='\033[1m[[name]]\033[0m = \033[94m{[[value]]:5.3f}\033[0m'):
        """
        Initialize the Callback.
        :param metrics_dict:            Dictionary containing mappings for metrics names/keys
                                        e.g. {"accuracy": "acc", "val. accuracy": "val_acc"}
        :param num_epochs:              Number of training epochs
        :param log_frequency:           Log frequency (in epochs)
        :param metric_string_template:  (opt.) String template to print each metric
        """
        super().__init__()

        self.metrics_dict = collections.OrderedDict(metrics_dict)
        self.num_epochs = num_epochs
        self.log_frequency = log_frequency

        # We build a format string to later print the metrics, (e.g. "Epoch 0/9: loss = 1.00; val-loss = 2.00")
        log_string_template = 'Epoch {0:2}/{1}: '
        separator = '; '

        i = 2
        for metric_name in self.metrics_dict:
            templ = metric_string_template.replace('[[name]]', metric_name).replace('[[value]]', str(i))
            log_string_template += templ + separator
            i += 1

        # We remove the "; " after the last element:
        log_string_template = log_string_template[:-len(separator)]
        self.log_string_template = log_string_template

    def on_train_begin(self, logs=None):
        print("Training: {}start{}".format(log_begin_red, log_end_format))

    def on_train_end(self, logs=None):
        print("Training: {}end{}".format(log_begin_green, log_end_format))

    def on_epoch_end(self, epoch, logs={}):
        if (epoch - 1) % self.log_frequency == 0 or epoch == self.num_epochs:
            values = [logs[self.metrics_dict[metric_name]] for metric_name in self.metrics_dict]
            print(self.log_string_template.format(epoch, self.num_epochs, *values))


# Callback to simply log metrics at the end of each epoch (saving space compared to verbose=1):
metrics_to_print = collections.OrderedDict([("loss", "loss"),
                                            ("v-loss", "val_loss"),
                                            ("acc", "acc"),
                                            ("v-acc", "val_acc"),
                                            ("top5-acc", "top5_acc"),
                                            ("v-top5-acc", "val_top5_acc")])

callback_simple_log = SimpleLogCallback(metrics_to_print,
                                        num_epochs=EPOCHS, log_frequency=2)

model_dir = './models/ResNet50_ImageNet'
callbacks = [
    # Callback to interrupt the training if the validation loss/metrics stops improving for some epochs:
    tf.keras.callbacks.EarlyStopping(patience=8, monitor='val_acc',
                                     restore_best_weights=True),
    # Callback to log the graph, losses and metrics into TensorBoard:
    tf.keras.callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0, write_graph=True),
    # Callback to save the model (e.g., every 5 epochs), specifying the epoch and val-loss in the filename:
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(model_dir, 'weights-epoch{epoch:02d}-loss{val_loss:.2f}.h5'), period=5),
    # Log callback:
    callback_simple_log
]

history = ResNet18.fit(train_dataset,
                       epochs=EPOCHS, steps_per_epoch=int(TRAIN_NUM_EXAMPLES/BATCH_SIZE),
                       validation_data=val_dataset, validation_steps=int(TRAIN_NUM_EXAMPLES/BATCH_SIZE),
                       verbose=1, callbacks=callbacks)


fig, ax = plt.subplots(3, 2, figsize=(15, 10), sharex='col') # add parameter `sharey='row'` for a more direct comparison
ax[0, 0].set_title("loss")
ax[0, 1].set_title("val-loss")
ax[1, 0].set_title("acc")
ax[1, 1].set_title("val-acc")
ax[2, 0].set_title("top5-acc")
ax[2, 1].set_title("val-top5-acc")

ax[0, 0].plot(history.history['loss'])
ax[0, 1].plot(history.history['val_loss'])
ax[1, 0].plot(history.history['acc'])
ax[1, 1].plot(history.history['val_acc'])
ax[2, 0].plot(history.history['top5_acc'])
ax[2, 1].plot(history.history['val_top5_acc'])


best_val_acc = max(history.history['val_acc']) * 100
best_val_top5 = max(history.history['val_top5_acc']) * 100

print('Best val acc:  {:2.2f}%'.format(best_val_acc))
print('Best val top5: {:2.2f}%'.format(best_val_top5))

