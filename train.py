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

config = configparser.ConfigParser()
config.read("config.ini")



def preprocess(image, segmentation):
    """
    A preprocess function that is runned after images are read.
    """
    # Setting variables to 'True' will make the program run selected operations. Only one variable can be true at a time
    rand_crop = False
    flip_up_down = False
    flip_left_right = False
    color_change = False
    # Setting Height and width values and starting point for boxes
    x_pic, y_pic = 224, 224
    x_min, y_min = 0, 0

    # Set images size to a constant
    image = tf.image.resize_images(image, [x_pic, y_pic])
    segmentation = tf.image.resize_images(segmentation, [x_pic, y_pic], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.to_float(image) / 255
    segmentation = tf.to_int64(segmentation)
    segmentation = tf.where(segmentation > 0, tf.ones_like(segmentation), tf.zeros_like(segmentation))
    # Doing some processing

    if rand_crop is True:
        # Random cropping
        box = np.ones([1, 1, 4])
        boxes_size = [y_min, x_min, x_pic - 1, y_pic - 1]
        for i in range(4):
            box[:, :, i] = boxes_size[i] / x_pic

        bbox = tf.convert_to_tensor(box, np.float32)
        begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
            tf.shape(image), bounding_boxes=bbox, min_object_covered=0.25,
            aspect_ratio_range=[0.75, 1.33], area_range=[0.05, 1.0], max_attempts=100,
            use_image_if_no_bounding_boxes=True)

        # Employ the bounding box to distort the image.
        image = tf.slice(image, begin, size)
        image = tf.image.resize_images(image, [x_pic, y_pic])
        image.set_shape([x_pic, y_pic, 3])

        segmentation = tf.slice(segmentation, begin, size)
        segmentation = tf.image.resize_images(segmentation, [x_pic, y_pic])
        segmentation.set_shape([x_pic, y_pic, 1])
        return image, segmentation

    elif flip_up_down is True:
        # Flipping up/down
        image = tf.image.random_flip_up_down(image, seed=25)
        segmentation = tf.image.random_flip_up_down(segmentation, seed=25)
        return image, segmentation

    elif flip_left_right is True:
        # Flipping left/right
        image = tf.image.random_flip_left_right(image, seed=30)
        segmentation = tf.image.random_flip_left_right(segmentation, seed=30)
        return image, segmentation

    elif color_change is True:
        # Color changes
        image = tf.image.random_hue(image, max_delta=0.3)
        return image, segmentation

    else:
        return image, segmentation


def filenames(dataset_folder, test=None):
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


def keras_model(input_shape):

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

    model.add(layers.Conv2D(1, 1, strides=(1, 1), padding='same', activation='sigmoid'))
    # Compile loss and optimizer
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model


def fetch_data(debug_mode=config['testing/debug'].getboolean('debug_mode'), test=None):
    """

    :return:
    """
    if test is True:
        image_names = filenames('data', test=True)
        x = []
        sess = tf.InteractiveSession()
        with tf.variable_scope('preprocess'):
            for i, img_path in enumerate(image_names):

                # Read image
                img = misc.imread(img_path)

                # Preprocess image
                x_pic, y_pic = 224, 224

                # Set images size to a constant
                img = tf.image.resize_images(img, [x_pic, y_pic])
                img = tf.to_float(img) / 255
                img = img.eval()

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
        sess = tf.InteractiveSession()
        with tf.variable_scope('preprocess'):
            for img_path, seg_path in tqdm(zip(image_names, segmentation_names)):

                # Read image
                img = misc.imread(img_path)
                seg = misc.imread(seg_path)

                # Preprocess image
                img, seg = preprocess(img, seg)
                img = img.eval()
                seg = seg.eval()[:, :, 0, None]

                x.append(img)
                y.append(seg)

                if debug_mode is True:
                    # Debug function
                    i += 1
                    if i == 10:
                        break
        return x, y


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

    #tbi_callback = TensorBoardImage('Image Example')

    return model, [tensorboard_callback, checkpoint]


def train_network():
    """

    :return:
    """


def split_data(x_data, y_data):
    """
    Splits data into training and validation
    :return:
    """
    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x_data, y_data, test_size=0.1,
                                                                              random_state=42)
    return np.array(x_train), np.array(x_val), np.array(y_train), np.array(y_val)


def main():
    # 'training' or 'testing' mode
    mode = config['testing/debug']['mode']
    print(mode)

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

        # If the dataset has gt labels, set this to <true>
        label = config['predictions'].getboolean('label')

        if label is True:
            # Fetch data set
            x_data, y_data = fetch_data(debug_mode=config['testing/debug'].getboolean('debug_mode'))

            # Split data
            x_train, x_val, y_train, y_val = split_data(x_data, y_data)

            # Load best weights from models
            model = models.load_model('models/weights.124-0.97.hdf5')
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
            model = models.load_model('models/weights.124-0.97.hdf5')
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
