import cv2
import numpy as np
import uuid
import os


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    median = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def color_seg(choice):
    if choice == 'blue':
        lower_hue = np.array([100, 30, 30])
        upper_hue = np.array([150, 148, 255])
    elif choice == 'white':
        lower_hue = np.array([0, 0, 0])
        upper_hue = np.array([0, 0, 255])
    elif choice == 'black':
        lower_hue = np.array([0, 0, 0])
        upper_hue = np.array([50, 50, 100])
    return lower_hue, upper_hue


def histogram_equalize(img):
    b, g, r = cv2.split(img)
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    return cv2.merge((blue, green, red))


def process_image(org_image, name=None):

    kernel = np.zeros((9, 9), np.float32)
    kernel[4, 4] = 2.0  # Identity, times two!
    box_filter = np.ones((9, 9), np.float32) / 81.0

    kernel = kernel - box_filter
    image = cv2.filter2D(org_image, -1, kernel)

    crop = histogram_equalize(image)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sensitivity = 10
    lower_white = np.array([0, 0, 255 - sensitivity])
    upper_white = np.array([255, sensitivity, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    image = cv2.bitwise_and(image, image, mask=mask)
    print(image.shape)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    image = cv2.cvtColor(thresh , cv2.COLOR_GRAY2BGR)

    image = cv2.resize(image, (500, 500))
    if name is None:
        cv2.imwrite(os.path.join('croped2',  str(uuid.uuid4())) + ".jpg", image)
    else:
        cv2.imwrite(os.path.join('croped2',  str(name).split("\\")[-1]) + ".jpg", image)
    return image
