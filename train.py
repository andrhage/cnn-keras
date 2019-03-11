from __future__ import print_function

import glob
import os
import numpy as np

import tensorflow as tf


def generator_for_filenames(*filenames):
    """
    Wrapping a list of filenames as a generator function
    """
    def generator():
        for f in zip(*filenames):
            yield f
    return generator


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


def read_image_and_segmentation(img_f, seg_f):
    """
    Read images from file using tensorflow and convert the segmentation to appropriate formate.
    :param img_f: filename for image
    :param seg_f: filename for segmentation
    :return: Image and segmentation tensors
    """
    img_reader = tf.read_file(img_f)
    seg_reader = tf.read_file(seg_f)
    img = tf.image.decode_png(img_reader, channels=3)
    seg = tf.image.decode_png(seg_reader)
    seg = tf.where(seg > 0, tf.ones_like(seg), tf.zeros_like(seg))
    return img, seg


def tensors_from_filenames(image_names, segmentation_names, preprocess=preprocess, batch_size=8):
    """
    Convert a list of filenames to tensorflow images.
    :param image_names: image filenames
    :param segmentation_names: segmentation filenames
    :param preprocess: A function that is run after the images are read, the takes image and
    segmentation as input
    :param batch_size: The batch size returned from the function
    :return: Tensors with images and corresponding segmentations
    """
    dataset = tf.data.Dataset.from_generator(
        generator_for_filenames(image_names, segmentation_names),
        output_types=(tf.string, tf.string),
        output_shapes=(None, None)
    )

    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.map(read_image_and_segmentation)
    dataset = dataset.map(preprocess)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    return dataset.repeat().make_one_shot_iterator().get_next()


def filenames(dataset_folder, training=True):
    sub_dataset = 'training' if training else 'testing'
    segmentation_names = glob.glob(os.path.join(dataset_folder, sub_dataset, 'gt', '*-ground_truth*.png'),
                                   recursive=True)
    image_names = glob.glob(os.path.join(dataset_folder, sub_dataset, 'images', '*-47-*.png'),
                                   recursive=True)
    return image_names, segmentation_names


def model(img, seg):
    """
    Improved model by adding more layers and cross entropy loss
    :param img:
    :param seg:
    :return:
    """
    # Setting variable to 'True' will make the program run selected operation
    l2_reg = False

    # Setting the l2-reg parameter
    l2 = tf.contrib.layers.l2_regularizer(0.001)

    if l2_reg is True:
        # Down sampling 4 layers encoding
        x = tf.layers.conv2d(img, 16, [5, 16], kernel_regularizer=l2,
                             strides=(2, 2), padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 16, [5, 16], kernel_regularizer=l2,
                             strides=(2, 2), padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 32, [5, 32], kernel_regularizer=l2,
                             strides=(2, 2), padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 32, [5, 32], kernel_regularizer=l2,
                             strides=(2, 2), padding='same', activation=tf.nn.relu)

        # Up sampling 4 layers decoding
        x = tf.layers.conv2d_transpose(x, 32, [5, 32], kernel_regularizer=l2,
                                       strides=(2, 2), padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, 32, [5, 32], kernel_regularizer=l2,
                                       strides=(2, 2), padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, 16, [5, 16], kernel_regularizer=l2,
                                       strides=(2, 2), padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, 16, [5, 16], kernel_regularizer=l2,
                                       strides=(2, 2), padding='same', activation=tf.nn.relu)

        x = tf.layers.conv2d(x, 1, 1, padding='same', activation=tf.nn.sigmoid)

        x = tf.image.resize_images(x, [224, 224])

        # Adding a l2 regularization
        cross_entropy = tf.to_float(seg) * tf.log(1e-3 + x) + (1 - tf.to_float(seg)) * tf.log((1 - x) + 1e-3)
        loss_reg = tf.losses.get_regularization_loss()
        loss = tf.reduce_mean(- cross_entropy) + loss_reg
        return x, loss
    else:
        # Down sampling 4 layers encoding
        x = tf.layers.conv2d(img, 16, [5, 16],
                             strides=(2, 2), padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 16, [5, 16],
                             strides=(2, 2), padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 32, [5, 32],
                             strides=(2, 2), padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 32, [5, 32],
                             strides=(2, 2), padding='same', activation=tf.nn.relu)

        # Up sampling 4 layers decoding
        x = tf.layers.conv2d_transpose(x, 32, [5, 32],
                                       strides=(2, 2), padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, 32, [5, 32],
                                       strides=(2, 2), padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, 16, [5, 16],
                                       strides=(2, 2), padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, 16, [5, 16],
                                       strides=(2, 2), padding='same', activation=tf.nn.relu)

        x = tf.layers.conv2d(x, 1, 1, padding='same', activation=tf.nn.sigmoid)

        x = tf.image.resize_images(x, [224, 224])

        # Adding a cross entropy loss function
        cross_entropy = tf.to_float(seg) * tf.log(1e-3 + x) + (1 - tf.to_float(seg)) * tf.log((1 - x) + 1e-3)
        loss = tf.reduce_mean(- cross_entropy)
        return x, loss


def tensorboard(img, img_val, seg_val, logits_val, logits, seg, loss, loss_val, step):
    """
    Setting up Tensorboard
    :param img:
    :param img_val:
    :param seg_val:
    :param logits_val:
    :param logits:
    :param seg:
    :param loss:
    :param loss_val:
    :param step:
    :return:
    """

    # Setting variables for Family in Tensorboard
    family_train = 'Training'
    family_val = 'Validation'

    # Setting up calculations for the validation summary
    accuracy_val, accuracy_update_val = tf.metrics.accuracy(seg_val, logits_val)
    precision_val, precision_update_val = tf.metrics.precision(seg_val, logits_val)
    recall_val, recall_update_val = tf.metrics.recall(seg_val, logits_val)
    # 𝛽 = 1
    f_measure_val = (2 * precision_update_val * recall_update_val) / (precision_update_val + recall_update_val)

    # Setting up calculations for the training summary
    accuracy_train, accuracy_update_train = tf.metrics.accuracy(seg, logits)
    precision_train, precision_update_train = tf.metrics.precision(seg, logits)
    recall_train, recall_update_train = tf.metrics.recall(seg, logits)
    # 𝛽 = 1
    f_measure_train = (2 * precision_update_train * recall_update_train) / (precision_update_train + recall_update_train)

    # Tensorboard scalars for training
    tf.summary.scalar('Accuracy', accuracy_update_train, family=family_train)
    tf.summary.scalar('Precision', precision_update_train, family=family_train)
    tf.summary.scalar('Recall', recall_update_train, family=family_train)
    tf.summary.scalar('F_measure', f_measure_train, family=family_train)
    tf.summary.scalar('Loss', loss, family=family_train)
    tf.summary.scalar('Step', step, family=family_train)

    # Tensorboard scalars for validation
    tf.summary.scalar('Accuracy', accuracy_update_val, family=family_val)
    tf.summary.scalar('Precision', precision_update_val, family=family_val)
    tf.summary.scalar('Recall', recall_update_val, family=family_val)
    tf.summary.scalar('F_measure', f_measure_val, family=family_val)
    tf.summary.scalar('Loss_val', loss_val, family=family_val)
    tf.summary.scalar('Step', step, family=family_val)

    # Making a zeros array for superimposed images
    zeros_train = tf.zeros(tf.shape(logits))

    # Tensorboard images training
    superimposed_img_train = img + tf.concat(axis=3, values=(zeros_train, zeros_train, logits))
    superimposed_img_train = tf.clip_by_value(superimposed_img_train, 0, 1)
    superimposed_img_seg_train = img + tf.concat(axis=3, values=(zeros_train, tf.to_float(seg), zeros_train))
    superimposed_img_seg_train = tf.clip_by_value(superimposed_img_seg_train, 0, 1)
    tf.summary.image('Training', img, max_outputs=1, family=family_train)
    tf.summary.image('Training_predicted', logits, max_outputs=1, family=family_train)
    tf.summary.image('Seg_training', tf.cast(seg, dtype=tf.float32), max_outputs=1, family=family_train)
    tf.summary.image('Superimposed_training', superimposed_img_train, max_outputs=1, family=family_train)
    tf.summary.image('Superimposed_training_label', superimposed_img_seg_train, max_outputs=1, family=family_train)

    # Making a zeros array for superimposed images
    zeros_val = tf.zeros(tf.shape(logits_val))

    # Tensorboard images validation
    superimposed_img_val = img_val + tf.concat(axis=3, values=(zeros_val, zeros_val, logits_val))
    superimposed_img_val = tf.clip_by_value(superimposed_img_val, 0, 1)
    superimposed_img_seg_val = img_val + tf.concat(axis=3, values=(zeros_val, tf.to_float(seg_val), zeros_val))
    superimposed_img_seg_val = tf.clip_by_value(superimposed_img_seg_val, 0, 1)
    tf.summary.image('Validation', img_val, max_outputs=1, family=family_val)
    tf.summary.image('Validation_predicted', logits_val, max_outputs=1, family=family_val)
    tf.summary.image('Seg_val', tf.cast(seg_val, dtype=tf.float32), max_outputs=1, family=family_val)
    tf.summary.image('Superimposed_validation', superimposed_img_val, max_outputs=1, family=family_val)
    tf.summary.image('Superimposed_validation_label', superimposed_img_seg_val, max_outputs=1, family=family_val)


def main(_):
    # Getting filenames from the kitti dataset
    image_names, segmentation_names = filenames('data')

    # Get image tensors from the filenames
    img, seg = tensors_from_filenames(
        image_names[:-3],
        segmentation_names[:-3],
        batch_size=8)
    # Get the validation tensors
    img_val, seg_val = tensors_from_filenames(
        image_names[-3:],
        segmentation_names[-3:],
        batch_size=8)

    #Create the model
    with tf.variable_scope('model'):
        logits, loss = model(img, seg)

    #Reuse the same model for validation
    with tf.variable_scope('model', reuse=True):
        logits_val, loss_val = model(img_val, seg_val)

    #Keep track of number of steps
    step = tf.train.get_or_create_global_step()
    #Create an optimizer
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = tf.train.AdamOptimizer(0.001).minimize(
            loss,
            global_step=step)

    # Setting up Tensorboard
    tensorboard(img, img_val, seg_val, logits_val, logits, seg, loss, loss_val, step)

    # tf.summary.scalar('step', step)

    saver_hook = tf.train.CheckpointSaverHook('logs', save_secs=15)
    summary_saver_hook = tf.train.SummarySaverHook(
        summary_op=tf.summary.merge_all(),
        output_dir='logs',
        save_secs=15
    )

    #Run training
    with tf.train.SingularMonitoredSession(
            hooks=[saver_hook, summary_saver_hook],
            checkpoint_dir='logs') as sess:
        while not sess.should_stop():
            _, loss_, step_ = sess.run([train_op, loss, step])

            print(step_, 'loss', loss_)
            if step_ % 10 == 0:
                loss_ = sess.run([loss_val])
                print('\t\t\tval_loss', loss_)


if __name__ == '__main__':
    main(None)
