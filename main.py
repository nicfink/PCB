# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import dippykit as dip
import numpy as np

def load_pcb(name):
    im = dip.imread(name)

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

    Gx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    Gy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

    #TODO figure how to convolve with the 2 1D component vectors of the sobel matrix
    #g1 = np.array([[1],
    #              [2],
    #              [1]])
    #g2 = np.array([1, 0, -1])

    print ("begin conv")
    sobel_im_x = dip.convolve2d(Gx, im_grey)
    sobel_im_y = dip.convolve2d(Gy, im_grey)
    print ("end conv")
    #find left edge
    for i in range(5,im_size[1]-5):
        j = center_y
        if abs(sobel_im_x[j,i]) > 0.2:
            left_edge = i
            print(left_edge)
            break
    #find right edge
    for i in range(im_size[1]-5, 5, -1):
        j = center_y
        if abs(sobel_im_x[j,i]) > 0.2:
            right_edge = i
            break
    # find top edge
    for i in range(5, im_size[0] - 5):
        j = center_x
        if abs(sobel_im_y[i, j]) > 0.2:
            top_edge = i
            break
    # find bottom edge
    for i in range(im_size[0] - 5, 5, -1):
        j = center_x
        if abs(sobel_im_y[i, j]) > 0.2:
            bottom_edge = i
            break


    im_trim = im_float[top_edge:bottom_edge, left_edge:right_edge, :]
    #dip.imshow(im_trim)
    #dip.show()
    detect_comps(im_trim)


def detect_comps(board):
    color = np.mean(board[10, :], axis = 0)
    print(np.shape(color))
    background = np.full((5, 5, 3), color)
    size = np.shape(board)
    #TODO find a better way to compare the difference between the background and the board
    for i in range(0, size[0]-4, 5):
        for j in range(0, size[1]-4, 5):
            dip.imshow(background[:,:,0])
            dip.show()
            if dip.SSIM(board[i:i+5, j:j+5, 0], background[:, :, 0])[0] > 90: #and dip.SSIM(board[i:i+5, j:j+5, 1], background[:, :, 1])[0] > 90 and dip.SSIM(board[i:i+5, j:j+5, 2], background[:, :, 2])[0] > 90:
                board[i:i+5, j:j+5] = [1, 1, 1]
    #dip.imshow(board)
    dip.show()


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




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    load_pcb("simplePCB.jpg")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
