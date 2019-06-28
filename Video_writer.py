import cv2
import os

# Put in directory location
image_folder = 'predictions/rapportklart_materiale/sequential_custom_2_transfer_exclusive/'
video_name = 'predictions/rapportklart_materiale/sequential_custom_2_transfer_exclusive/sequential_custom_2_transfer_exclusive.avi'
#images = os.listdir(image_folder)
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort(key=lambda x: int(x[:][0:-4]))
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
