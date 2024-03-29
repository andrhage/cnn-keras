import glob
import numpy as np
import os
import cv2
from scipy import misc
from statistics import mean


def filenames(dataset_folder):
    """
    Returning a list with image names from both testing and training directory according to users choice

    :param dataset_folder:
    :param test:
    :return:
    """
    sub_dataset = 'accuracy'
    gt_names = glob.glob(os.path.join(dataset_folder, sub_dataset, 'gt', '**.png'),
                                           recursive=True)
    prediction_names = glob.glob(os.path.join(dataset_folder, sub_dataset, 'prediction', '**.png'),
                                           recursive=True)
    return prediction_names, gt_names


def main():
    # Set to True if those datasets is used
    freiburg = False

    # Read images
    prediction_names, gt_names = filenames('data')
    accuracy_list, c_22_list, c_11_list, c_12_list, c_21_list, IoU_1_list, IoU_2_list, mIoU_list = [], [], [], [], [], [], [], []
    i = 1
    # Go through each image
    for prediction_path, gt_path in zip(prediction_names, gt_names):

        # Read image
        prediction = misc.imread(prediction_path)
        gt = misc.imread(gt_path)

        # Resize image to 224x224
        prediction = cv2.resize(prediction, (224, 224))
        gt = cv2.resize(gt, (224, 224))
        # Cast to float
        prediction = prediction.astype(float)
        gt = gt.astype(float)

        # Normalizing
        prediction = np.array((prediction - np.min(prediction)) / (np.max(prediction) - np.min(prediction)))
        gt = np.array((gt - np.min(gt)) / (np.max(gt) - np.min(gt)))

        # Execute only if its a freiburg dataset
        if freiburg is True:
            # Mask out other classes than road and background, and converts the image
            bg = gt[:, :, 0] == gt[:, :, 1]  # B == G
            gr = gt[:, :, 1] == gt[:, :, 2]  # G == R
            gt = np.bitwise_and(bg, gr, dtype=np.uint8)
            gt = np.repeat(gt[..., None], 1, axis=2)
            prediction = prediction[:, :, 0, None]

            # Setup confusion matrix
            c_11 = np.sum((gt == 1) * (prediction == 1))
            c_12 = np.sum((gt == 1) * (prediction == 0))
            c_21 = np.sum((gt == 0) * (prediction == 1))
            c_22 = np.sum((gt == 0) * (prediction == 0))

            # Setting up the metrics
            accuracy = (c_11 + c_22) / (c_11 + c_22 + c_12 + c_21)
            IoU_1 = c_11 / (c_11 + c_12 + c_21)
            IoU_2 = c_22 / (c_22 + c_21 + c_12)
            mIoU = (IoU_1 + IoU_2) / 2

            print('**********************************************************')
            print('{} {}'.format('Accuracy : ', accuracy))
            print('---------------------------------------------------------------')
            print('{} {}'.format('IoU Road: ', IoU_1))
            print('---------------------------------------------------------------')
            print('{} {}'.format('IoU Background: ', IoU_2))
            print('---------------------------------------------------------------')
            print('{} {}'.format('mIoU : ', mIoU))
            print('---------------------------------------------------------------')
            print(i)
            i += 1
            print('**********************************************************')


        # Normal gt and prediction images
        else:
            # Gt with red road
            gt = gt[:, :, 0, None]
            prediction = prediction[:, :, 0, None]

            # Setup confusion matrix
            c_11 = np.sum((gt == 1) * (prediction == 1))
            c_12 = np.sum((gt == 1) * (prediction == 0))
            c_21 = np.sum((gt == 0) * (prediction == 1))
            c_22 = np.sum((gt == 0) * (prediction == 0))

            # Setting up the metrics
            accuracy = (c_11 + c_22) / (c_11 + c_22 + c_12 + c_21)
            IoU_1 = c_11 / (c_11 + c_12 + c_21)
            IoU_2 = c_22 / (c_22 + c_21 + c_12)
            mIoU = (IoU_1 + IoU_2) / 2
            print('**********************************************************')
            print('{} {}'.format('Accuracy : ', accuracy))
            print('---------------------------------------------------------------')
            print('{} {}'.format('IoU Road: ', IoU_1))
            print('---------------------------------------------------------------')
            print('{} {}'.format('IoU Background: ', IoU_2))
            print('---------------------------------------------------------------')
            print('{} {}'.format('mIoU : ', mIoU))
            print('---------------------------------------------------------------')
            print(i)

            i += 1
            print('**********************************************************')
        # Append accuracies to lists
        accuracy_list.append(accuracy)
        c_22_list.append(c_22)
        c_11_list.append(c_11)
        c_12_list.append(c_12)
        c_21_list.append(c_21)
        IoU_1_list.append(IoU_1)
        IoU_2_list.append(IoU_2)
        mIoU_list.append(mIoU)
        if freiburg is True:
            # Removes 'nan' from all three lists
            accuracy_list = [i for i in accuracy_list if str(i) != 'nan']
            IoU_1_list = [i for i in IoU_1_list if str(i) != 'nan']
            IoU_2_list = [i for i in IoU_2_list if str(i) != 'nan']
            mIoU_list = [i for i in mIoU_list if str(i) != 'nan']

    sum_accuracy = mean(accuracy_list)
    sum_IoU_1 = mean(IoU_1_list)
    sum_IoU_2 = mean(IoU_2_list)
    sum_mIoU = mean(mIoU_list)
    sum_tn = mean(c_22_list)
    sum_tp = mean(c_11_list)
    sum_fp = mean(c_12_list)
    sum_fn = mean(c_21_list)
    print('Accuracy: {} IoU_1: {} IoU_2: {} mIoU: {} TN: {} TP: {} FP: {} FN: {}'
          .format(sum_accuracy, sum_IoU_1, sum_IoU_2, sum_mIoU, sum_tn, sum_tp, sum_fp, sum_fn))


if __name__ == '__main__':
    main()
