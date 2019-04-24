from PIL import Image
import glob


for filename in glob.iglob('/Masteroppgave/Master-thesis/data/testing/images/**/*.jpg', recursive=True):
    print(filename[0:-4])
    im = Image.open(filename)
    im.save(filename[0:-4] + '.png')
