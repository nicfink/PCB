import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import glob
import dippykit as dip



def loader():
    library_path = './pcb_dataset/*'
    new_path = './dataset/'
    dict = {'capacitor': 0, 'diode': 1, 'ic': 2, 'inductor': 3, 'resistor': 4, 'transistor': 5,
            'capacitors': 0, 'diodes': 1, 'ICs': 2, 'inductors':3, 'resistors':4, 'transistors':5}
    training_set = []
    for f in glob.iglob(library_path):
        i = dip.imread(f)
        name = f.split('\\')[1].split('_')[0]
        label = dict[name]
        image = cv2.resize(dip.float_to_im(i, 8), (100, 100))
        dip.im_write(image, f)
        image = image.reshape((3, 100, 100))
        tens = torch.Tensor(image)
        lab = torch.LongTensor([label])
        tup = (tens, lab)
        training_set.append(tup)

    return training_set

classes = ('capacitor', 'diode', 'ic', 'inductor', 'resistor', 'transistor')