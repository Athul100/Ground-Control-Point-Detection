import cv2


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)

        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        min_rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(min_rect)
        import  numpy as np
        box = np.int0(box)
        x1 = abs(box[0][0] - box[1][0])
        x2 = abs(box[1][0] - box[2][0])
        x3 = abs(box[2][0] - box[3][0])
        y1 = abs(box[0][1] - box[1][1])
        y2 = abs(box[1][1] - box[2][1])
        y3 = abs(box[2][1] - box[3][1])

        x = max(x1,x2,x3)
        y = max(y1,y2,y3)

        x_min = min(x1, x2, x3)
        y_min = min(y1, y2, y3)

        print(box)
        print(x,y)

        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area
        print("solidity ", solidity)

        if x > y:
            ratio = x /y
        else:
            ratio = y/x
        print("ratio" , ratio)
        if 1 <= ratio < 2 and 0.5 < solidity < 0.9:
            shape = "could be L shapeed"

        # return the name of the shape
        return shape, x_min, y_min, x, y
