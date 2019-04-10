import glob
import os

os.makedirs('gt')
for filename in glob.iglob('Sekvens2_png/**/*label.png', recursive=True):
    print(filename)
    file = filename.split('/')[2]
    folder = filename.split('/')[1]
    print(file)
    print(folder)
    os.replace(filename, 'gt/' + folder + '_gt' + '.png')

