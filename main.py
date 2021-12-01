# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import dippykit as dip
import numpy as np
import cv2

import os
import glob
import matplotlib.pyplot as plt
from string import digits


def load_pcb(name):


    im = dip.imread(name) #read in the PCB


    im_float = dip.im_to_float(im) #Convert it to float [0, 1]
    im_grey = np.mean(im_float, axis=2) #greyscale
    im_size = np.shape(im_grey)  #get the size

    #Find the center x and y positions to help detect the edges of the board
    center_x = round(im_size[1] / 2)
    center_y = round(im_size[0] / 2)


    #sobel edge detection
    left_edge = 0
    right_edge = im_size[0]
    top_edge = 0
    bottom_edge = im_size[1]


    sobel_im_x = dip.edge_detect(im_grey, "sobel_v")
    sobel_im_y = dip.edge_detect(im_grey, "sobel_h")

    # Don't go to the exact edges of the image because the sobel edge detection finds edges near the edge of the image.
    # Start a little inwards
    # Find Left Edge
    for i in range(5,im_size[1]-5):
        j = center_y
        if abs(sobel_im_x[j,i]) > 0.1:
            left_edge = i
            print(left_edge)
            break
    #find right edge
    for i in range(im_size[1]-5, 5, -1):
        j = center_y
        if abs(sobel_im_x[j,i]) > 0.1:
            right_edge = i
            break
    # find top edge
    for i in range(5, im_size[0] - 5):
        j = center_x
        print(i)
        if abs(sobel_im_y[i, j]) > 0.1:
            top_edge = i
            break
    # find bottom edge
    for i in range(im_size[0] - 5, 5, -1):
        j = center_x
        if abs(sobel_im_y[i, j]) > 0.1:
            bottom_edge = i
            break

    # Crop the image to roughly these edges
    im_trim = im_float[top_edge:bottom_edge, left_edge:right_edge, :]
    dip.imshow(im_trim)
    dip.show()
    #Remove the background from the image without removing components
    im_noback = get_layer(dip.float_to_im(im_trim), layer="fore", K = 3)

    im_back = get_layer(dip.float_to_im(im_trim), layer="back", K = 3)

    #im_noback = get_layer(im_noback, layer="min", K = 3)

    #Find regions of interest on the board

    roi = group_by_contours(im_noback, threshold=20) #find regions by their contours
    upscale_fact = 4 #for upscaling
    res_matrix = np.array([[1/upscale_fact, 0],
                           [0, 1/upscale_fact]])
    #For each region of interest, show a picture
    for i in range(0, np.shape(roi)[0]):
        comp = im_trim[roi[i][0]:roi[i][1], roi[i][2]:roi[i][3]]
        upscale_comp = dip.resample(comp, res_matrix, interpolation="bicubic")
        ## TODO check for image recognition
        ########################Element Identification######
        #library_path = ("drive/My Drive/DIP/library_components/*")
        ##Import the element that was cropped from the pcb board
        #element_id = element_identification(upscale_comp, library_path)
        #print(element_id)
        ########################Element Identification######
        if np.shape(comp)[0] > 100:
            roi2 = group_by_black_space(im_noback[roi[i][0]:roi[i][1], roi[i][2]:roi[i][3]], threshold=20) #use black space sorting to check this area
            for j in range(0, np.shape(roi2)[0]): #for each subregion of interest
                comp = im_trim[roi[i][0] + roi2[j][0]:roi[i][0] + roi2[j][1],
                                roi[i][2] + roi2[j][2]:roi[i][2] + roi2[j][3]] #cut out each subregion this region
                upscale_comp = dip.resample(comp, res_matrix, interpolation="bicubic") #upscale each subregion
                ##TODO check the subregion for recognition
                dip.imshow(upscale_comp)
                dip.show()
        else:
            dip.imshow(upscale_comp)
            dip.show()


#layer is what we want to see, back = background, fore = foreground, min = smallest
def get_layer(board: np.array, layer: str = "fore", K: int = 3): #K is the number of groups we're finding
    board = cv2.cvtColor(board,cv2.COLOR_RGB2HSV) #Convert to HSV, works better this way
    vectorized_board = board.reshape(-1, 3) #convert to a vector for the kmeans analysis
    vectorized_board = np.float32(vectorized_board) #convert to float which is required
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1) #criteria

    attempts = 10 #k means has 10 attempts to identify the areas best

    #we only care about the labels, which identify which elements belong to which group
    ret, label, center = cv2.kmeans(vectorized_board, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)


    #get the group/label names and how many elements are in each group
    vals, counts = np.unique(label, return_counts=True)

    ind1 = np.argmax(counts) #largest group
    ind2 = np.argmin(counts) #smallest group
    back_cluster = vals[ind1] #background group (assuming background is the largest group, which isn't *always* true)
    min_cluster = vals[ind2] #smallest group

    masked_image = np.copy(board)
    bum = cv2.cvtColor(masked_image, cv2.COLOR_HSV2RGB)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((board.shape))
    dip.imshow(res2)
    dip.show()
    # convert to the shape of a vector of pixel values
    masked_image = masked_image.reshape((-1, 3))
    #either enable or disable different layers (ex foreground is everything but background)
    if layer == "fore":
        masked_image[label.flatten() == back_cluster] = [0, 0, 0]
    elif layer == "min":
        masked_image[label.flatten() != min_cluster] = [0, 0, 0]
    else:
        masked_image[label.flatten() != back_cluster] = [0, 0, 0]
    # convert back to original shape
    masked_image = masked_image.reshape(board.shape)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_HSV2RGB)
    dip.imshow(masked_image)
    dip.show()
    return masked_image



def group_by_contours(noback_im, threshold: int = 10):

    grey_im = np.mean(noback_im, axis=2) #convert to greyscale

    #lowpas filtering
    #blur = np.array([[1/9, 1/9, 1/9],
    #                 [1/9, 1/9, 1/9],
    #                 [1/9, 1/9, 1/9]])

    #grey_im = dip.convolve2d(grey_im, blur) #blur the image gently

    roi = [] #regions of interest
    grey_im = cv2.convertScaleAbs(grey_im) #this is needed for the contour mapping, not sure why exactly

    #Does a bit of empirically determined thresholding to better identify contours
    thresh = 20
    ret, grey_im = cv2.threshold(grey_im, thresh, 255, cv2.THRESH_BINARY)
    #grey_im[grey_im > 0] = 255
    #dip.imshow(grey_im, 'gray')
    #dip.figure()


    contours, hierarchy = cv2.findContours(grey_im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Finds the contours in the image
    ##create an empty image for contours
    img_contours = np.zeros(grey_im.shape)
    ##draw the contours on the empty image
    cv2.drawContours(img_contours, contours, -1, 255, 3)
    dip.imshow(img_contours)
    dip.show()

    #for each contour, create a bounding box. If it is large enough, then return it
    for cnt in contours:
        cnt = cnt.reshape(-1, 2)
        max_y = np.max(cnt[:, 1])
        min_y = np.min(cnt[:, 1])
        max_x = np.max(cnt[:, 0])
        min_x = np.min(cnt[:, 0])
        if max_y-min_y > threshold and max_x-min_x > threshold:
            roi.append([min_y, max_y, min_x, max_x])
    return roi












def group_by_black_space(noback_im, threshold: int = 25):
    grey_im = np.mean(noback_im, axis=2) #greyscale

    #blur the image
    blur = np.array([[1 / 9, 1 / 9, 1 / 9],
                     [1 / 9, 1 / 9, 1 / 9],
                     [1 / 9, 1 / 9, 1 / 9]])
    grey_im = dip.convolve2d(grey_im, blur)
    roi = []

    for i in range (0, np.shape(grey_im)[0]):
        for j in range (0, np.shape(grey_im)[1]):
            if grey_im[i,j] != 0: #Find a nonzero pixel
                y_min = i
                y_max = y_min
                while y_max < np.shape(grey_im)[0] and grey_im[y_max, j] != 0: #iterate in the y direction until you reach zero pixel
                    y_max += 1
                y_mid = int((y_max+y_min)/2)
                x_min = j
                x_max = j
                #from the y midpoint
                while x_max < np.shape(grey_im)[1] and grey_im[y_mid, x_max] != 0: #iterate forward in x direction until you reach zero pixel
                    x_max += 1
                while x_min >= 0 and grey_im[y_mid, x_min] != 0:#iterate backward in x direction until you reach zero pixel
                    x_min -= 1
                # construct a bounding box and set pixels in box to zero
                # if box is big enough, then add it as a region of interest
                if x_max - x_min > threshold and y_max - y_min > threshold:
                    grey_im[y_min + 1:y_max, x_min + 1:x_max] = 0
                    roi.append([y_min+1, y_max, x_min+1, x_max])
                    dip.show()
                #if box not big enough, then just set the current pixel to zero,
                # other pixels in the current box may be valuable later
                else:
                    grey_im[i, j] = 0
    return roi

########################Element Identification######
def element_identification(test_path, library_path):
    im = test_path  # read in the test component
    original = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    M, N = original.shape
    # dip.imshow(original, 'gray')
    # dip.show()
    original = cv2.normalize(original, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # Load all the images
    all_images_to_compare = []
    titles = []
    ssim_index = []
    for f in glob.iglob(library_path):
        image = cv2.imread(f)
        titles.append(f)
        all_images_to_compare.append(image)

    # dip.imshow(all_images_to_compare[1])

    i = 0
    maxdict = {}

    for image_to_compare, title in zip(all_images_to_compare, titles):
        compare_image = all_images_to_compare[i]
        compare_image = cv2.cvtColor(compare_image, cv2.COLOR_BGR2GRAY)
        comp_im_reshaped = cv2.resize(compare_image, (N, M), interpolation=cv2.INTER_LINEAR)

        ##NOTE: The component is identified by the title of the image in which it is similar
        ##NOTE: library images are required to have the name of the component
        title = os.path.basename(title)
        title = title[:title.rfind(".")]
        remove_digits = str.maketrans('', '', digits)
        title = title.translate(remove_digits)

        # print("Title: " + title)
        # dip.imshow(comp_im_reshaped,'gray')
        # dip.show()
        # calculates SSIM of the images
        ssim = dip.metrics.SSIM_contrast(original, comp_im_reshaped)
        flag = 0
        if i > 0:
            for p in maxdict.keys():

                if title == p:
                    flag = 1
                    if ssim[0] > maxdict.get(p):
                        maxdict[p] = ssim[0]
                        # print("Iam here")
        if flag == 0:
            maxdict.update({title: ssim[0]})

        # print(ssim[0])
        i += 1
    ##arranges the values to see the best comparison
    sorted_values = sorted(maxdict.values(), reverse=True)  # Sort the values
    sorted_dict = {}
    for i in sorted_values:
        for k in maxdict.keys():
            if maxdict[k] == i:
                sorted_dict[k] = maxdict[k]
                break
    # print(sorted_dict)
    # print("compenent is a :", list(sorted_dict.keys())[0])
    # print("SSIM value:", list(sorted_dict.values())[0])
    return (list(sorted_dict.keys())[0])


########################Element Identification######


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    load_pcb("simplePCB_ADI.jpg")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/






#im_float[:, :, 1] = np.zeros([im.shape[0], im.shape[1]])

    #im_float = np.mean(im_float, axis=2)

    #sin(theta/2) = sqrt(1-cos(theta)/2)


    #hpf = np.array([[-1, -1, -1],
    #                [-1, 8, -1],
    #               [-1, -1, -1]])

    #hpf = np.array([[.17, .67, .17],
    #                [.67, -3.5, .67],
    #                [.17, .67, .17]])


    #fft = dip.fft2(hpf)
    #dip.imshow(abs(fft))
    #im_filt = dip.convolve2d(im_float, hpf)
