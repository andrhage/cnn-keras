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

# Set to True if those datasets is used
kitti = False
freiburg = True

# Read images
prediction_names, gt_names = filenames('data')
x, y, z = [], [], []
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

    # Normalizing the values to between 0 and 1
    prediction = np.array((prediction - np.min(prediction)) / (np.max(prediction) - np.min(prediction)))
    gt = np.array((gt - np.min(gt)) / (np.max(gt) - np.min(gt)))

    # Set values to either 1 or 0 values
    # prediction = np.where(prediction > 0.5, np.ones_like(prediction),
    #                       np.zeros_like(prediction))
    # gt = np.where(gt > 0.5, np.ones_like(gt), np.zeros_like(gt))

    # Execute only if its a kitti dataset
    if kitti is True:
        # Gt with pink/blue road
        gt = gt[:, :, 2, None]
        prediction = prediction[:, :, 0, None]
    # Execute only if its a freiburg dataset
    elif freiburg is True:
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

        IoU_1 = c_11 / (c_11 + c_12 + c_21)
        IoU_2 = c_22 / (c_22 + c_21 + c_12)
        mIoU = (IoU_1 + IoU_2) / 2
        print('**********************************************************')
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

        IoU_1 = c_11 / (c_11 + c_12 + c_21)
        IoU_2 = c_22 / (c_22 + c_21 + c_12)
        mIoU = (IoU_1 + IoU_2) / 2
        print('**********************************************************')
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
    x.append(IoU_1)
    y.append(IoU_2)
    z.append(mIoU)
    if freiburg is True:
        x = [i for i in x if str(i) != 'nan']
        y = [i for i in y if str(i) != 'nan']
        z = [i for i in z if str(i) != 'nan']

tot_IoU_1 = mean(x)
tot_IoU_2 = mean(y)
tot_mIoU = mean(z)
print('1: {} 2: {} 3: {}'.format(tot_IoU_1, tot_IoU_2, tot_mIoU))
