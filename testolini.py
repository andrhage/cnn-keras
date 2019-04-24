from __future__ import print_function
from PIL import Image
import glob
import os
import numpy as np
from scipy import misc
import sklearn.model_selection
import sklearn
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
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import h5py
from keras import applications

config = configparser.ConfigParser()
config.read("config.ini")


def fetch_data(debug_mode=config['train/test/debug'].getboolean('debug_mode'), test=None):
    """
    Reading the images and prepares them for training/testing before appending them in a list

    :return:
    """
    if test is True:
        image_names = filenames('data', test=True)
        x = []
        for i, img_path in tqdm(enumerate(image_names)):

            # Read image
            img = misc.imread(img_path)

            # Preprocess image
            img = cv2.resize(img, (int(config['data_processing']['x_pic']),
                                   int(config['data_processing']['y_pic'])))
            img = img.astype(float) / 255
            # Data augmentation under here:

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
            img = cv2.resize(img, (int(config['data_processing']['x_pic']),
                                   int(config['data_processing']['y_pic'])))
            seg = cv2.resize(seg, (int(config['data_processing']['x_pic']),
                                   int(config['data_processing']['y_pic'])))
            img = img.astype(float) / 255
            seg = seg.astype(float)
            seg = np.array((seg - np.min(seg)) / (np.max(seg) - np.min(seg)))

            if config['data_processing'].getboolean('kitti') is True:
                # Gt with pink/blue road
                seg = seg[:, :, 2, None]
            elif config['data_processing'].getboolean('freiburg') is True:
                bg = seg[:, :, 0] == seg[:, :, 1]  # B == G
                gr = seg[:, :, 1] == seg[:, :, 2]  # G == R
                seg = np.bitwise_and(bg, gr, dtype=np.uint8)
                seg = np.repeat(seg[..., None], 1, axis=2)
            else:
                # Gt with red road
                seg = seg[:, :, 0, None]


            # if augmentation:
            #     aug_img = aug(img)
            #     aug_seg = aug_seg(seg)
            #     x.append(aug_img)
            #     y.append(aug_seg)

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


def split_data(x_data, y_data):
    """
    Splits data into training and validation sets

    :return:
    """
    x_train, x_val, y_train, y_val = \
        sklearn.model_selection.train_test_split(x_data,
                                                 y_data,
                                                 test_size=float(config['train/test/debug']['test_size']),
                                                 random_state=42)
    return np.array(x_train), np.array(x_val), np.array(y_train), np.array(y_val)


def keras_model_sequential(input_shape):
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
    # Output layer
    model.add(layers.Conv2D(1, 1, strides=(1, 1), padding='same', activation='sigmoid'))
    # Compile loss and optimizer, and printing the network structure summary
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model


def keras_model_residual():
    """
    Based on "https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277"
    :param input_shape:
    :return:
    """

    # U-Net model
    inputs = Input((int(config['data_processing']['x_pic']), int(config['data_processing']['y_pic']), 3))
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def setup_model_and_tensorboard(x_train):
    """
    Sets up or loads a model
    :return:
    """
    if config['Network'].getboolean('sequential') is True:
        model = keras_model_sequential(x_train[0].shape)

        tensorboard_callback = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=8,
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


def cca_algorithm(prediction):
    """
    # The cca is based on this tutorial:
    # https://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/
    :param prediction:
    :return:
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
            mask = cv2.add(mask, labelmask)
            mask = ndi.binary_fill_holes(mask)
    return mask


def predict(x_test, image_names, model):
    """

    :param x_test:
    :param image_names:
    :param model:
    :return:
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
        # Processing with connected component analysis
        mask = cca_algorithm(prediction)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(np.squeeze(prediction))
        ax2.imshow(np.squeeze(img))

        # With or without cca
        if config['predictions'].getboolean('cca') is True:
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
    # 'training' or 'testing' mode
    mode = config['train/test/debug']['mode']

    if mode == 'training':
        # Fetch data set
        x_data, y_data = fetch_data(debug_mode=config['train/test/debug'].getboolean('debug_mode'))

        # Split data
        x_train, x_val, y_train, y_val = split_data(x_data, y_data)

        # Choosing whether to use data augmentation or not
        if config['train/test/debug'].getboolean('augmentation') is True:

            # Create model
            model, callbacks_model = setup_model_and_tensorboard(x_train)

            data_generator = ImageDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)

            data_generator.fit(x_train)

            model.fit_generator(data_generator.flow(x_train, y_train, batch_size=10),
                                steps_per_epoch=len(x_train) / 10, epochs=int(config['train/test/debug']['epochs']),
                                validation_data=(x_val, y_val), shuffle=True, verbose=1, callbacks=callbacks_model
                                )

        else:
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
                # model_res = applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None,
                #                                  input_shape=(224, 224, 3), pooling=None, classes=1000)
                # for layer in model_res.layers:
                #     layer.trainable = False
                # model_res.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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

            # Predicting the output
            predict(x_test, image_names, model)


if __name__ == '__main__':
    main()
