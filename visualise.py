import cv2
import json
from process import process_image, image_resize
import uuid


def visualise_bounding_boxes(image, coordinates):
    print("visualising data {} with coordinates {}".format(image, coordinates))
    coordinate_object = json.loads(coordinates)
    frame = cv2.imread(image)
    h, w, _ = frame.shape
    # frame = cv2.resize(frame, (500, 375), interpolation= cv2.INTER_NEAREST )
    frame = process_image(frame)

    h1, w1 = frame.shape

    scale_h = h / h1
    scale_w = w / w1
    cv2.circle(frame, (round(coordinate_object[0][0] / scale_w), round(coordinate_object[0][1] / scale_h)), 5,
               (255, 255, 255), 1)
    cv2.imshow("Display 1", frame)
    cv2.waitKey(0)

