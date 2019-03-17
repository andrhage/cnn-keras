from __future__ import print_function

import glob
import os
import numpy as np

from scipy import misc
import sklearn.model_selection
import sklearn
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import models
from keras import layers
from keras import callbacks
from tqdm import tqdm
import configparser
import cv2


config = configparser.ConfigParser()
config.read("config.ini")


def preprocess(image, segmentation):
    """
    Preprocessing images. User can choose of the following preprocessing operations:
    1. Random crops
    2. Flip up down
    3. Flip left right
    4. Color changes
    """

    # Doing some processing

    if config['data_processing'].getboolean('rand_crop') is True:
        # Random cropping
        box = np.ones([1, 1, 4])
        boxes_size = [int(config['data_processing']['y_min']), int(config['data_processing']['x_min']),
                      int(config['data_processing']['x_pic']) - 1, int(config['data_processing']['y_pic']) - 1]
        for i in range(4):
            box[:, :, i] = boxes_size[i] / int(config['data_processing']['x_pic'])

        bbox = tf.convert_to_tensor(box, np.float32)
        begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
            tf.shape(image), bounding_boxes=bbox, min_object_covered=0.25,
            aspect_ratio_range=[0.75, 1.33], area_range=[0.05, 1.0], max_attempts=100,
            use_image_if_no_bounding_boxes=True)

        # Employ the bounding box to distort the image.
        image = tf.slice(image, begin, size)
        image = tf.image.resize_images(image, [int(config['data_processing']['x_pic']),
                                               int(config['data_processing']['y_pic'])])
        image.set_shape([int(config['data_processing']['x_pic']), int(config['data_processing']['y_pic']), 3])

        segmentation = tf.slice(segmentation, begin, size)
        segmentation = tf.image.resize_images(segmentation, [int(config['data_processing']['x_pic']), int(config['data_processing']['y_pic'])])
        segmentation.set_shape([int(config['data_processing']['x_pic']), int(config['data_processing']['y_pic']), 1])
        return image, segmentation

    elif config['data_processing'].getboolean('flip_up_down') is True:
        # Flipping up/down
        image = tf.image.random_flip_up_down(image, seed=25)
        segmentation = tf.image.random_flip_up_down(segmentation, seed=25)
        return image, segmentation

    elif config['data_processing'].getboolean('flip_left_right') is True:
        # Flipping left/right
        image = tf.image.random_flip_left_right(image, seed=30)
        segmentation = tf.image.random_flip_left_right(segmentation, seed=30)
        return image, segmentation

    elif config['data_processing'].getboolean('color_change') is True:
        # Color changes
        image = tf.image.random_hue(image, max_delta=0.3)
        return image, segmentation

    else:
        return image, segmentation


def fetch_data(debug_mode=config['testing/debug'].getboolean('debug_mode'), test=None):
    """
    Reading the images and prepares them for training/testing before appending them in a list

    :return:
    """
    if test is True:
        image_names = filenames('data', test=True)
        x = []
        for i, img_path in enumerate(image_names):

            # Read image
            img = misc.imread(img_path)

            # Preprocess image
            img = cv2.resize(img, (int(config['data_processing']['x_pic']), int(config['data_processing']['y_pic'])))
            img = img.astype(float) / 255

            x.append(img)

            if debug_mode is True:
                # Debug function
                i += 1
                if i == 10:
                    break
        return x
    else:
        image_names, segmentation_names = filenames('data')
        x, y = [], []
        i = 0
        for img_path, seg_path in tqdm(zip(image_names, segmentation_names)):

            # Read image
            img = misc.imread(img_path)
            seg = misc.imread(seg_path)

            # Preprocess image
            # Set images size to a constant
            img = cv2.resize(img, (int(config['data_processing']['x_pic']), int(config['data_processing']['y_pic'])))
            seg = cv2.resize(seg, (int(config['data_processing']['x_pic']), int(config['data_processing']['y_pic'])))
            img = img.astype(float) / 255
            seg = seg.astype(int)
            seg = np.array((seg - np.min(seg)) / (np.max(seg) - np.min(seg)))
            seg = seg[:, :, 0, None]

            x.append(img)
            y.append(seg)

            if debug_mode is True:
                # Debug function
                i += 1
                if i == 10:
                    break
        return x, y


def filenames(dataset_folder, test=None):
    """
    Returning a list with image names from both testing and training directory according to users choice

    :param dataset_folder:
    :param test:
    :return:
    """
    if test is True:
        sub_dataset = 'testing'
        image_names = glob.glob(os.path.join(dataset_folder, sub_dataset, 'images', '*-47-*.png'),
                                recursive=True)
        return image_names
    else:

        sub_dataset = 'training'
        segmentation_names = glob.glob(os.path.join(dataset_folder, sub_dataset, 'gt', '*-ground_truth*.png'),
                                           recursive=True)
        image_names = glob.glob(os.path.join(dataset_folder, sub_dataset, 'images', '*-47-*.png'),
                                           recursive=True)
        return image_names, segmentation_names


def split_data(x_data, y_data):
    """
    Splits data into training and validation sets

    :return:
    """
    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x_data, y_data, test_size=0.1,
                                                                              random_state=42)
    return np.array(x_train), np.array(x_val), np.array(y_train), np.array(y_val)


def keras_model(input_shape):
    """
    The conv net model implemented with Keras

    :param input_shape:
    :return:
    """
    # Initializing
    model = models.Sequential()
    # Input layer
    model.add(layers.Conv2D(32, 11, strides=(2, 2), padding='same', activation='relu', input_shape=input_shape))
    # Conv layers
    model.add(layers.Conv2D(16, 7, strides=(2, 2), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, 5, strides=(2, 2), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, 5, strides=(2, 2), padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(32, 5, strides=(2, 2), padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(32, 5, strides=(2, 2), padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(16, 5, strides=(2, 2), padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(16, 5, strides=(2, 2), padding='same', activation='relu'))
    # Activation layer
    model.add(layers.Conv2D(1, 1, strides=(1, 1), padding='same', activation='sigmoid'))
    # Compile loss and optimizer, and printing the network structure summary
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model


def setup_model_and_tensorboard(x_train):
    """
    Sets up or loads a model
    :return:
    """
    model = keras_model(x_train[0].shape)

    tensorboard_callback = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=8, write_graph=True,
                                                 write_grads=True, write_images=True,
                                                 embeddings_freq=0, embeddings_layer_names=None,
                                                 embeddings_metadata=None, embeddings_data=None,
                                                 update_freq='epoch'
                                                 )
    checkpoint = callbacks.ModelCheckpoint('models/weights.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc',
                                           verbose=1, save_best_only=True,
                                           save_weights_only=False, mode='auto', period=1)

    return model, [tensorboard_callback, checkpoint]


def main():
    # 'training' or 'testing' mode
    mode = config['testing/debug']['mode']

    if mode == 'training':
        # Fetch data set
        x_data, y_data = fetch_data(debug_mode=config['testing/debug'].getboolean('debug_mode'))

        # Split data
        x_train, x_val, y_train, y_val = split_data(x_data, y_data)

        # Create model
        model, callbacks_model = setup_model_and_tensorboard(x_train)

        # Train model
        models.Sequential.fit(model, x_train, y_train, batch_size=8,
                              epochs=350, verbose=1, validation_data=(x_val, y_val),
                              shuffle=True, callbacks=callbacks_model
                              )

    elif mode == 'testing':

        label = config['predictions'].getboolean('label')

        if label is True:
            # Fetch data set
            x_data, y_data = fetch_data(debug_mode=config['testing/debug'].getboolean('debug_mode'))

            # Split data
            x_train, x_val, y_train, y_val = split_data(x_data, y_data)

            # Load best weights from models
            model = models.load_model('models/' + config['predictions']['weights'])
            model.summary()

            for img, seg in zip(x_val, y_val):
                img = np.reshape(img, (1, 224, 224, 3))
                prediction = models.Sequential.predict(model, img)
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                ax1.imshow(np.squeeze(prediction))
                ax2.imshow(np.squeeze(img))
                ax3.imshow(np.squeeze(seg))
                plt.show()
        else:
            # Fetch test data set
            x_test = fetch_data(debug_mode=config['testing/debug'].getboolean('debug_mode'), test=True)

            # Load best weights from models
            model = models.load_model('models/' + config['predictions']['weights'])
            model.summary()
            for i, img in enumerate(x_test):
                img = np.reshape(img, (1, 224, 224, 3))
                prediction = models.Sequential.predict(model, img)
                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(np.squeeze(prediction))
                ax2.imshow(np.squeeze(img))
                fig.savefig(('{}/prediction_' + str(i) + '.png').format('D:/Masteroppgave/Master-thesis/predictions'))
                plt.show()
                i += 1


if __name__ == '__main__':
    main()
