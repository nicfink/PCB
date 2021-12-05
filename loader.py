import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import glob
import dippykit as dip
import numpy as np



def loader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    library_path = './pcb_dataset/*'
    new_path = './dataset/'
    dict = {'capacitor': 0, 'diode': 1, 'ic': 2, 'inductor': 3, 'resistor': 4, 'transistor': 2}
    dict2 = {'capacitors': 0, 'diodes': 1, 'ICs': 2, 'inductors':3, 'resistors':4, 'transistors':2}
    training_set = []
    for f in glob.iglob(library_path): #Get all the sub folders
        for im in glob.iglob(f+'\*'):
            i = dip.imread(im)
            name = im.split('\\')[2].split('_')[0]
            label = dict2[name]
            image = cv2.resize(dip.float_to_im(i, 8), (100, 100))
            dip.im_write(image, im)
            #image = image.reshape((3, 32, 32))
            tens = torch.Tensor(image)
            tens_im = transform(image)
            lab = torch.LongTensor([label])
            tup = (tens_im, lab)
            training_set.append(tup)

    np.random.shuffle(training_set)
    return training_set

classes = ('capacitor', 'diode', 'ic', 'inductor', 'resistor', 'transistor')