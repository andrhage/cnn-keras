import glob
import numpy as np
import os
import cv2
from scipy import misc


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


kitti = False
freiburg = False
prediction_names, gt_names = filenames('data')
x, y = [], []
i = 0
for prediction_path, gt_path in zip(prediction_names, gt_names):

    # Read image
    prediction = misc.imread(prediction_path)
    gt = misc.imread(gt_path)

    prediction = cv2.resize(prediction, (224, 224))
    gt = cv2.resize(gt, (224, 224))
    prediction = prediction.astype(float) / 255
    gt = gt.astype(float)
    gt = np.array((gt - np.min(gt)) / (np.max(gt) - np.min(gt)))

    if kitti is True:
        # Gt with pink/blue road
        gt = gt[:, :, 2, None]
    elif freiburg is True:
        # Mask out other classes than road and background, and converts the image
        bg = gt[:, :, 0] == gt[:, :, 1]  # B == G
        gr = gt[:, :, 1] == gt[:, :, 2]  # G == R
        gt = np.bitwise_and(bg, gr, dtype=np.uint8)
        gt = np.repeat(gt[..., None], 1, axis=2)
    else:
        # Gt with red road
        gt = gt[:, :, 0, None]

    x.append(prediction)
    y.append(gt)

c_11 = np.sum((y == 1) * (x == 1))
c_12 = np.sum((y == 1) * (x == 0))
c_21 = np.sum((y == 0) * (x == 1))
c_22 = np.sum((y == 0) * (x == 0))
print(c_11)
print('-------------------------------------------------')
print(c_12)
print('-------------------------------------------------')
print(c_21)
print('-------------------------------------------------')
print(c_22)
