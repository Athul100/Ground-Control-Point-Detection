import cv2
from model import ModelClass
import numpy as np
from create_dataset import  preprocess_input
from process import process_image
from findblobs import detect_shape
import pandas as pd
import os

path_to_image = r'C:\Users\dj\Downloads\athul_workspace\python_codes\fullimage\DJI_0037.JPG.jpg'
path_to_weights = r'mode;-110-478.1933.hdf5'

im = cv2.imread(path_to_image)
print(im.shape)

height, width, _ = im.shape
padding = 10

area = 500

mode = ModelClass()
model = mode.load_model(path_to_weights)

x1 = 0
y1 = 0
x2 = area
y2 = area

net_data = width / area
net_height = height / area
prediction_coordinates = []

for i in range(int(net_data)):
    py1 = 0

    py2 = area

    for h in range(int(net_height)):
        x_pad = 0
        if x1 - padding >= 0:
            x_pad = padding
        y_pad = 0
        if py1 - padding >= 0:
            y_pad = padding

        crop = im[py1 - y_pad:py2, x1 - x_pad:x2]
        img = process_image(crop)
        image_data = preprocess_input(img)
        data_set = np.ndarray(shape=(1, 500, 500, 3), dtype=np.float32)

        data_set[0, :, :, :] = image_data
        prediction = model.predict(data_set)
        coordinate = []
        for value in prediction[0]:
            coordinate.append(round(float(value) + 10))

        lis = np.array_split(coordinate, 3)

        cord = detect_shape(img)

        for l in lis:
            r = 30
            x = round(l[0])
            y = round(l[1] )
            error_rate = 10
            if x < 50 and y <50:
                continue

            cv2.circle(im, ((i * area) + round(l[0]), (h * area) + round(l[1])), 30,
                       (255, 255, 255), 1)
            prediction_coordinates.append([(i * area) + round(l[0]), (h * area) + round(l[1])])

        print(lis)

        py1 = py1 + area
        py2 = py2 + area
    x1 = x1 + area
    x2 = x2 + area

cv2.imwrite(os.path.join('predicted', 'pred.jpg'), im)

a = [path_to_image, prediction_coordinates]
my_df = pd.DataFrame(a)

my_df.to_csv('my_csv.csv', index=False, header=False)
