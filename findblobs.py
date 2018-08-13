import cv2
from shape import ShapeDetector
import imutils


def detect_shape(ord_image):

    resized = imutils.resize(ord_image, width=300)
    ratio = ord_image .shape[0] / float(resized.shape[0])

    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    sd = ShapeDetector()
    num = 0
    contours = []
    for c in cnts:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        print(ratio)
        try:
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
        except:
            continue
        shape, x_min, y_min,x, y= sd.detect(c)

        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        if shape == 'could be L':
            contours.append({
                'x_min' : x_min,
                'y_min' : y_min
            })
            cv2.drawContours(ord_image, [c], -1, (0, 255, 0), 2)
            # cv2.imshow("Image", ord_image)
            # cv2.waitKey(0)
    return contours
