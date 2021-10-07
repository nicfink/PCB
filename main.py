# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import dippykit as dip
import numpy as np
import cv2

def load_pcb(name):
    im = dip.imread(name)
    #detect_comps(im)
    im_float = dip.im_to_float(im)
    im_grey = np.mean(im_float, axis=2)
    im_size = np.shape(im_grey)
    center_x = round(im_size[1] / 2)
    center_y = round(im_size[0] / 2)
    #sobel edge detection
    left_edge = 0
    right_edge = im_size[0]
    top_edge = 0
    bottom_edge = im_size[1]

    #Gx = np.array([[1, 0, -1],
    #               [2, 0, -2],
    #               [1, 0, -1]])
    #Gy = np.array([[1, 2, 1],
    #               [0, 0, 0],
    #               [-1, -2, -1]])

    #print ("begin conv")
    sobel_im_x = dip.edge_detect(im_grey, "sobel_v")
    sobel_im_y = dip.edge_detect(im_grey, "sobel_h")

    #sobel_im_y = dip.convolve2d(Gy, im_grey)
    #print ("end conv")
    #find left edge
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


    im_trim = im_float[top_edge:bottom_edge, left_edge:right_edge, :]
    #dip.imshow(im_trim)
    #dip.show()
    im_noback = detect_comps(dip.float_to_im(im_trim))
    roi = group(im_noback)
    print (roi)
    for i in range(0, np.shape(roi)[0]):
        dip.imshow(im_trim[roi[i][0]:roi[i][1], roi[i][2]:roi[i][3]])
        dip.show()


def detect_comps(board):
    #dip.imshow(board)
    #dip.show()

    vectorized_board = board.reshape(-1, 3)
    vectorized_board = np.float32(vectorized_board)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
    K = 4 #anecdotally, this seems to make the most sense as 3 or 4
    attempts = 10
    ret, label, center = cv2.kmeans(vectorized_board, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    print(label.flatten())
    res = center[label.flatten()]
    result_image = res.reshape((board.shape))
    dip.imshow(result_image)
    dip.show()


    #for cluster in range(0, K):
    #    masked_image = np.copy(board)
    #    # convert to the shape of a vector of pixel values
    #    masked_image = masked_image.reshape((-1, 3))
    #    # color (i.e cluster) to disable
    #    masked_image[label.flatten() == cluster] = [0, 0, 0]
    #    # convert back to original shape
    #    masked_image = masked_image.reshape(board.shape)
    #    # show the image
    #    dip.imshow(masked_image)
    #    dip.show()
    #Assuming the background layer has the most elements in it, we try to find the largest layer:

    vals, counts = np.unique(label, return_counts=True)
    ind = np.argmax(counts)
    back_cluster = vals[ind]
    masked_image = np.copy(board)
    # convert to the shape of a vector of pixel values
    masked_image = masked_image.reshape((-1, 3))
    # color (i.e cluster) to disable
    masked_image[label.flatten() == back_cluster] = [0, 0, 0]
    # convert back to original shape
    masked_image = masked_image.reshape(board.shape)
    # show the image
    #dip.imshow(masked_image)
    #dip.show()
    return masked_image


def group(noback_im):
    grey_im = np.mean(noback_im, axis=2)
    blur = np.array([[1/9, 1/9, 1/9],
                     [1/9, 1/9, 1/9],
                     [1/9, 1/9, 1/9]])
    #dip.imshow(grey_im, 'gray')
    #dip.figure()
    grey_im = dip.convolve2d(grey_im, blur)
    #dip.imshow(grey_im, 'gray')
    #dip.show()
    #print(np.shape(grey_im)[0])
    roi = []
    for i in range (0, np.shape(grey_im)[0]):
        for j in range (0, np.shape(grey_im)[1]):
            #print(i)
            #print(j)
            if grey_im[i,j] != 0:
                y_min = i

                y_max = y_min
                while y_max < np.shape(grey_im)[0] and grey_im[y_max, j] != 0:
                #    print(i)
                #    print(j+k)
                    y_max += 1
                y_mid = int((y_max+y_min)/2)
                #print(y_min)
                #print(y_max)
                #print(y_mid)
                x_min = j
                x_max = j
                while x_max < np.shape(grey_im)[1] and grey_im[y_mid, x_max] != 0:
                    x_max += 1
                while x_min >= 0 and grey_im[y_mid, x_min] != 0:
                    x_min -= 1
                #print(x_min)
                #print(x_max)
                #print(y_min)
                #print(y_max)

                if x_max - x_min > 20 and y_max - y_min > 20:
                    grey_im[y_min + 1:y_max, x_min + 1:x_max] = 0
                    roi.append([y_min+1, y_max, x_min+1, x_max])
                    dip.show()
                else:
                    grey_im[i, j] = 0
    return roi




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
