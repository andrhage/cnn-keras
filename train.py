from __future__ import print_function

import glob
import os
import random
import numpy as np
from scipy import misc
import sklearn.model_selection
import sklearn
from skimage import color
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import callbacks
from tqdm import tqdm
import configparser
import cv2
from skimage import measure
from scipy import ndimage as ndi
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

# Sets up the parser environment which let's the user change hyperparameters in this code through config.ini
config = configparser.ConfigParser()
config.read("config.ini")


def filenames(dataset_folder, test=None):
    """
    Returning a list with image names from both testing and training directory according to users choice

    :param dataset_folder:
    :param test: None
    :return: image_names, segmentation_names
    """
    if test is True:
        sub_dataset = 'testing'
        image_names = glob.glob(os.path.join(dataset_folder, sub_dataset, 'images', '**.png'),
                                recursive=True)
        return image_names
    else:

        sub_dataset = 'training'
        segmentation_names = glob.glob(os.path.join(dataset_folder, sub_dataset, 'gt', '**.png'),
                                           recursive=True)
        image_names = glob.glob(os.path.join(dataset_folder, sub_dataset, 'images', '**.png'),
                                           recursive=True)
        return image_names, segmentation_names


def fetch_data(debug_mode=config['train/test/debug'].getboolean('debug_mode'), test=None):
    """
    This function includes two different tasks.
    The first is to read data from directory sort it and append their filenames in list.
    The second is to execute some pre-proccessing of the data, like resizing, normalizing and/or data augmentation

    :return: x, y
    """
    # Test refers to test images
    if test is True:
        image_names = filenames('data', test=True)
        x = []
        for i, img_path in tqdm(enumerate(image_names)):

            # Read image
            img = misc.imread(img_path)

            # Preprocess image
            img = cv2.resize(img, (int(config['data_processing']['x_pic']),
                                   int(config['data_processing']['y_pic'])))
            # Normalize
            img = img.astype(float) / 255

            x.append(img)

            # Debug mode takes only 10 images from the dataset-directory to save time during debug
            if debug_mode is True:
                # Debug function
                i += 1
                if i == 10:
                    break
        return x
    # This runs in training mode
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
            img = cv2.resize(img, (int(config['data_processing']['x_pic']),
                                   int(config['data_processing']['y_pic'])))
            seg = cv2.resize(seg, (int(config['data_processing']['x_pic']),
                                   int(config['data_processing']['y_pic'])))
            # Normalize
            img = img.astype(float) / 255
            seg = seg.astype(float)
            seg = np.array((seg - np.min(seg)) / (np.max(seg) - np.min(seg)))

            if config['data_processing'].getboolean('kitti') is True:
                # Gt with pink/blue road
                seg = seg[:, :, 2, None]
            elif config['data_processing'].getboolean('freiburg') is True:
                # Mask out other classes than road and background, and converts the image
                bg = seg[:, :, 0] == seg[:, :, 1]  # B == G
                gr = seg[:, :, 1] == seg[:, :, 2]  # G == R
                seg = np.bitwise_and(bg, gr, dtype=np.uint8)
                seg = np.repeat(seg[..., None], 1, axis=2)
            else:
                # Gt with red road
                seg = seg[:, :, 0, None]

            # Data augmentation
            if config['data_processing'].getboolean('flip') is True:
                if random.uniform(0, 1) > 0.5:
                    flip_img = cv2.flip(img, 1)
                    flip_seg = cv2.flip(seg, 1)
                    flip_seg = np.reshape(flip_seg, (int(config['data_processing']['x_pic']),
                                                     int(config['data_processing']['y_pic']), 1))
                    x.append(flip_img)
                    y.append(flip_seg)
            elif config['data_processing'].getboolean('color_change') is True:
                if random.uniform(0, 1) > 0.5:
                    color_change_img = color.convert_colorspace(img, 'RGB', 'RGB CIE')
                    x.append(color_change_img)
                    y.append(seg)

            x.append(img)
            y.append(seg)

            if debug_mode is True:
                # Debug function
                i += 1
                if i == 10:
                    break
        return x, y


def split_data(x_data, y_data):
    """
    Splits data into training and validation sets

    :return: 4 arrays where x_ refers to the original images and y_ to the ground truth
    """
    x_train, x_val, y_train, y_val = \
        sklearn.model_selection.train_test_split(x_data,
                                                 y_data,
                                                 test_size=float(config['train/test/debug']['test_size']),
                                                 random_state=42)
    return np.array(x_train), np.array(x_val), np.array(y_train), np.array(y_val)


def setup_model_and_tensorboard(x_train):
    """
    Sets up or loads a model in order to visualize the training live in tensorboard
    One setup for each network
    :return:
    """
    if config['Network'].getboolean('sequential') is True:
        model = keras_model_sequential(x_train[0].shape)

        tensorboard_callback = callbacks.TensorBoard(log_dir='./logs', histogram_freq=10, batch_size=10,
                                                     write_graph=True,
                                                     write_grads=True, write_images=True,
                                                     embeddings_freq=0, embeddings_layer_names=None,
                                                     embeddings_metadata=None, embeddings_data=None,
                                                     update_freq='epoch'
                                                     )
        checkpoint = callbacks.ModelCheckpoint('models/weights.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc',
                                               verbose=1,
                                               save_best_only=config['train/test/debug'].getboolean('save_best'),
                                               save_weights_only=False, mode='auto', period=1)

        return model, [tensorboard_callback, checkpoint]

    elif config['Network'].getboolean('residual') is True:
        model_res = keras_model_residual()

        tensorboard_callback = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=10,
                                                     write_graph=True,
                                                     write_grads=True, write_images=True,
                                                     embeddings_freq=0, embeddings_layer_names=None,
                                                     embeddings_metadata=None, embeddings_data=None,
                                                     update_freq='epoch'
                                                     )
        checkpoint_res = callbacks.ModelCheckpoint('models/weights.{epoch:02d}-{val_acc:.2f}.hdf5',
                                                   verbose=1,
                                               save_best_only=config['train/test/debug'].getboolean('save_best'),
                                               save_weights_only=False, mode='auto', period=1)

        # early_stop = EarlyStopping(patience=5, verbose=1)

        return model_res, [tensorboard_callback, checkpoint_res]


def keras_model_sequential(input_shape):
    """
    The conv net model implemented with Keras

    :param input_shape:
    :return:
    """
    # Initializing
    model = models.Sequential()
    # Input layer
    model.add(layers.Conv2D(16, 5, strides=(2, 2), padding='same', activation='relu', input_shape=input_shape))
    # Conv layers
    model.add(layers.Conv2D(16, 5, strides=(2, 2), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, 5, strides=(2, 2), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, 5, strides=(2, 2), padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(32, 5, strides=(2, 2), padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(32, 5, strides=(2, 2), padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(16, 5, strides=(2, 2), padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(16, 5, strides=(2, 2), padding='same', activation='relu'))
    # Output layer
    model.add(layers.Conv2D(1, 1, strides=(1, 1), padding='same', activation='sigmoid'))
    # Compile loss and optimizer, and printing the network structure summary
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model


def keras_model_residual():
    """
    Based on the following implementation:
    "https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277"

    :param input_shape:
    :return: model
    """

    # U-Net model
    inputs = Input((int(config['data_processing']['x_pic']), int(config['data_processing']['y_pic']), 3))
    conv_1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    conv_1 = Dropout(0.1)(conv_1)
    conv_1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_1)
    pool_1 = MaxPooling2D((2, 2))(conv_1)

    conv_2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_1)
    conv_2 = Dropout(0.1)(conv_2)
    conv_2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_2)
    pool_2 = MaxPooling2D((2, 2))(conv_2)

    conv_3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_2)
    conv_3 = Dropout(0.2)(conv_3)
    conv_3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_3)
    pool_3 = MaxPooling2D((2, 2))(conv_3)

    conv_4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_3)
    conv_4 = Dropout(0.2)(conv_4)
    conv_4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_4)
    pool_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

    conv_5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_4)
    conv_5 = Dropout(0.3)(conv_5)
    conv_5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_5)

    up_6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv_5)
    up_6 = concatenate([up_6, conv_4])
    conv_6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up_6)
    conv_6 = Dropout(0.2)(conv_6)
    conv_6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_6)

    up_7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (conv_6)
    up_7 = concatenate([up_7, conv_3])
    conv_7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up_7)
    conv_7 = Dropout(0.2)(conv_7)
    conv_7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_7)

    up_8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv_7)
    up_8 = concatenate([up_8, conv_2])
    conv_8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up_8)
    conv_8 = Dropout(0.1)(conv_8)
    conv_8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_8)

    up_9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv_8)
    up_9 = concatenate([up_9, conv_1], axis=3)
    conv_9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up_9)
    conv_9 = Dropout(0.1)(conv_9)
    conv_9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv_9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def cca_algorithm(prediction):
    """
    The cca algorithm is based on this tutorial:
    https://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/

    :param prediction:
    :return: mask
    """
    cca = measure.label(prediction, connectivity=None, background=0)
    mask = np.zeros(prediction.shape, dtype="uint8")
    for labels in np.unique(cca):
        if labels == 0:
            continue
        labelmask = np.zeros(prediction.shape, dtype="uint8")
        labelmask[cca == labels] = 255
        count_pixels = cv2.countNonZero(labelmask)
        if count_pixels > int(config['predictions']['threshold']):
            try:
                mask = cv2.add(mask, labelmask)
                mask = ndi.binary_fill_holes(mask)
            except TypeError:
                continue
    return mask


def predict(x_test, image_names, model):
    """
    This function executes several forms of data processing in order to optimize
    the predictions from each network even further.
    Processing:
    1. Thresholding
    2. Opening (Morphological operations)
    3. Connected component analysis and binary hole filling

    Then the function plots the predictions in matplotlib, before showing and saving the images

    :param x_test:
    :param image_names:
    :param model:
    """
    i = 0
    for img, names in zip(x_test, image_names):

        img = np.reshape(img, (1, int(config['data_processing']['x_pic']),
                               int(config['data_processing']['y_pic']), 3))

        # Predicts the road
        prediction = model.predict(img)
        # Filter out values with less certainty than 65 %
        prediction = np.where(prediction > 0.65, np.ones_like(prediction),
                              np.zeros_like(prediction))
        prediction = np.squeeze(prediction)
        # Processing with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        prediction = cv2.erode(prediction, kernel, iterations=3)
        prediction = cv2.dilate(prediction, kernel, iterations=2)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(np.squeeze(prediction))
        ax2.imshow(np.squeeze(img))

        # With or without cca
        if config['predictions'].getboolean('cca') is True:
            # Processing with connected component analysis
            mask = cca_algorithm(prediction)
            ax1.imshow(mask)
            fig.savefig(('{}/' + str(names[20:-4]) +
                         '.png').format('D:/Masteroppgave/Master-thesis/predictions'))
            if config['predictions'].getboolean('plots') is True:
                plt.show(block=False)
                plt.pause(0.5)
            i += 1
            plt.close()

            mask = np.reshape(mask, (int(config['data_processing']['x_pic']),
                                     int(config['data_processing']['y_pic']), 1)) * 255.0
            dump = np.zeros([np.shape(mask)[0], np.shape(mask)[1], 2])
            mask = np.concatenate((mask, dump), 2)
            misc.imsave(('{}/' + str(names[20:-4]) +
                         '.png').format('D:/Masteroppgave/Master-thesis/predictions/gt'), mask)
        else:
            ax1.imshow(prediction)

            fig.savefig(('{}/' + str(names[20:-4]) +
                         '.png').format('D:/Masteroppgave/Master-thesis/predictions'))
            if config['predictions'].getboolean('plots') is True:
                plt.show(block=False)
                plt.pause(0.5)
            i += 1
            plt.close()

            prediction = np.reshape(prediction, (int(config['data_processing']['x_pic']),
                                                 int(config['data_processing']['y_pic']), 1)) * 255.0
            dump = np.zeros([np.shape(prediction)[0], np.shape(prediction)[1], 2])
            prediction = np.concatenate((prediction, dump), 2)
            misc.imsave(('{}/' + str(names[20:-4]) +
                         '.png').format('D:/Masteroppgave/Master-thesis/predictions/gt'),
                        prediction)


def main():
    # 'training' or 'testing' mode. Choose string value for "mode" in config.ini
    mode = config['train/test/debug']['mode']

    if mode == 'training':
        # Fetch data set
        x_data, y_data = fetch_data(debug_mode=config['train/test/debug'].getboolean('debug_mode'))

        # Split data
        x_train, x_val, y_train, y_val = split_data(x_data, y_data)
        if config['Network'].getboolean('sequential') is True:
            # Create model
            model, callbacks_model = setup_model_and_tensorboard(x_train)

            # Train model
            models.Sequential.fit(model, x_train, y_train, batch_size=10,
                                  epochs=int(config['train/test/debug']['epochs']),
                                  verbose=1, validation_data=(x_val, y_val),
                                  shuffle=True, callbacks=callbacks_model
                                  )
        elif config['Network'].getboolean('residual') is True:
            # Create model
            model_res, callbacks_model_res = setup_model_and_tensorboard(x_train)

            # Transfer learning
            if config['train/test/debug'].getboolean('transfer') is True:
                model_res = load_model('models/' + config['train/test/debug']['weights'])
                for layer in model_res.layers[:int(config['train/test/debug']['layers'])]:
                    layer.trainable = False
            # Train model
            model_res.fit(x_train, y_train, batch_size=10,
                          epochs=int(config['train/test/debug']['epochs']),
                          shuffle=True, validation_data=(x_val, y_val),
                          callbacks=callbacks_model_res)

    elif mode == 'testing':

        label = config['predictions'].getboolean('label')

        if label is True:
            # Fetch data set
            x_data, y_data = fetch_data(debug_mode=config['train/test/debug'].getboolean('debug_mode'))

            # Split data
            x_train, x_val, y_train, y_val = split_data(x_data, y_data)

            # Load best weights from models
            model = models.load_model('models/' + config['predictions']['weights'])
            model.summary()

            for img, seg in zip(x_val, y_val):
                img = np.reshape(img, (1, int(config['data_processing']['x_pic']),
                                       int(config['data_processing']['y_pic']), 3))
                prediction = models.Sequential.predict(model, img)
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                ax1.imshow(np.squeeze(prediction))
                ax2.imshow(np.squeeze(img))
                ax3.imshow(np.squeeze(seg))
                plt.show()
        else:
            # Fetch test data set
            x_test = fetch_data(debug_mode=config['train/test/debug'].getboolean('debug_mode'), test=True)

            # Grab all the image names from each folder
            image_names = filenames('data', test=True)

            # Connect each image with its name and run through them one by one
            if config['Network'].getboolean('sequential') is True:
                # Load best weights from models
                model = models.load_model('models/' + config['predictions']['weights'])

            elif config['Network'].getboolean('residual') is True:
                model = load_model('models/' + config['predictions']['weights'])

            # Predicting the output and plots/saves it as images
            predict(x_test, image_names, model)


if __name__ == '__main__':
    main()
