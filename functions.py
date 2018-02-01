import numpy as np
import cv2
import math
from matplotlib import pyplot as plt


def create_histogram(img):
    hist = [0] * 256
    width, height = img.shape[:2]
    for row in range(width):
        for col in range(height):
            hist[img[row, col]] += 1

    return hist


def plot_histogram(hist, figure):
    plt.figure(figure)
    plt.plot(hist)
    plt.xlim([0, 255])


def linear_contrast_stretch(img):
    pmin = -1;
    pmax = -1;
    width, height = img.shape[:2]
    imgout = img.copy()

    hist = create_histogram(img)

    count = 0
    for gl in range(256):
        count+=hist[gl]
        if count >= height*width*0.01 and pmin<0:
            pmin = gl
        if count >= height*width*0.99 and pmax<0:
            pmax = gl

    for row in range(width):
        for col in range(height):
            pin = img[row,col]
            pout = (pin-pmin) * 255 / (pmax - pmin)
            if pout < 0:
                pout=0
            if pout > 255:
                pout=255
            imgout[row,col]=pout

    return imgout


def gamma_correction(img, r):
    width, height = img.shape[:2]
    imgout = img.copy()

    for row in range(width):
        for col in range(height):
            pin = img[row, col]
            pout = pow(255,1-r)*pow(pin, r)
            if pout < 0:
                pout = 0
            if pout > 255:
                pout = 255
            imgout[row,col] = pout

    return imgout


def equalization(img):
    width, height = img.shape[:2]
    imgout = img.copy()
    histogram = create_histogram(img)

    px = [0] * 256
    sumpx = [0] * 256

    for i in range(256):
        hi = histogram[i];
        px[i]= hi / (width * height);

        for k in range(i):
            sumpx[i] += px[k];

    for row in range(width):
        for col in range(height):
            pin = img[row,col]
            pout = 255 * sumpx[pin];
            if pout < 0:
                pout=0
            if pout > 255:
                pout = 255

            imgout[row,col] = pout
    return imgout


def convolution(img, kernel, k):
    width, height = img.shape[:2]
    imgout = img.copy()

    for row in range(width-k):
        for col in range(height-k):
            pout = 0
            # kerneling
            for me in range(k+1):
                m = me-k
                for ne in range(k+1):
                    n = ne-k
                    pout += img[row-m, col-n]*kernel[me][ne]
            # saturation
            if pout < 0:
                pout = 0
            if pout > 255:
                pout = 255

            imgout[row, col] = pout

    #imgout_enha = gamma_correction(imgout,0.6)
    return imgout


def angle_between(p1, p2):
    return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))


def show(img,imge):
    cv2.imshow("Original", img)
    cv2.imshow("Enhanced", imge)
    cv2.waitKey(0)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plot_histogram(hist, 1)
    hist = cv2.calcHist([imge], [0], None, [256], [0, 256])
    plot_histogram(hist, 2)
    plt.show()


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (w, h) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M,(h, w))

def addMeasure(img1,img2):
    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst
    return img1
