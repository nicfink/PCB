import glob
import dippykit as dip
import cv2

library_path = './pcb_dataset/*'
new_path = './dataset/'
dict = {'capacitor': 0, 'diode': 1, 'ic': 2, 'inductor': 3, 'resistor': 4, 'transistor': 5}
training_set = []
for f in glob.iglob(library_path):
    i = dip.imread(f)
    name = f.split('/')[2].split('_')[0]
    label = dict[name]
    image = cv2.resize(dip.float_to_im(i, 8), (100, 100))
    image = image.reshape((3, 100, 100))
    dip.im_write(image, f)
    tup = (image, label)
    training_set.append(tup)