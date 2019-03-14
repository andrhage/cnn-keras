import cv2

img = cv2.imread('D:/Masteroppgave/Andreas/data/training/gt/2017-05-09-13-26-47-00001-ground_truth.png')
print(img.shape)
print(img[:,:,:,None].shape)