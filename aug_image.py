
import cv2
import json
import copy
import os
padding = 10
folder_name = r"croped\\"


def check_if_coordinate_in_crop(x1, y1, x2, y2, coordinates):
    coordinate_arr = []
    for each_coordinate in coordinates:
        px = each_coordinate[0]
        py = each_coordinate[1]
        print(x1, px, x2, y1, py, y2)
        if x1 < px < x2 and y1 < py < y2:
            coordinate_arr.append([px-x1, py-y1])
    return coordinate_arr


def augment_image(image, dict_of_image_names_with_its_gcp_cordinates, new_cord,file_name):
    height, width, _ = image.shape
    horizontal_img = cv2.flip(image, 0)
    horizontal_cord = copy.deepcopy(new_cord)
    vertical_cord = copy.deepcopy(new_cord)
    both_cord = copy.deepcopy(new_cord)

    for each_cord in horizontal_cord:
        each_cord[1] = height - each_cord[1]

    filename = os.path.join(folder_name, file_name + '_horizontal.jpg')
    cv2.imwrite(filename, horizontal_img)
    dict_of_image_names_with_its_gcp_cordinates[filename] = horizontal_cord
    cv2.circle(horizontal_img , (round(int(horizontal_cord[0][0])), round(int(horizontal_cord[0][1]))), 15,
               (255, 255, 255), 1)

    vertical_img = cv2.flip(image, 1)

    for each_cord in vertical_cord:
        each_cord[0] = width - each_cord[0]

    filename = os.path.join(folder_name, file_name + '_vertical.jpg')
    cv2.imwrite(filename, vertical_img)
    dict_of_image_names_with_its_gcp_cordinates[filename] = vertical_cord

    cv2.circle(vertical_img, (round(int(vertical_cord[0][0])), round(int(vertical_cord[0][1]))), 15,
               (255, 255, 255), 1)

    both_img = cv2.flip(image, -1)

    for each_cord in both_cord:
        each_cord[0] = width - each_cord[0]
        each_cord[1] = height - each_cord[1]

    filename = os.path.join(folder_name, file_name + '_both.jpg')
    cv2.imwrite(filename, both_img)
    dict_of_image_names_with_its_gcp_cordinates[filename] = both_cord

    cv2.circle(both_img,  (round(int(both_cord[0][0])), round(int(both_cord[0][1]))), 15,
               (255, 255, 255), 1)
    '''
    cv2.imshow("Horizontal Flip", horizontal_img )
    cv2.imshow("Vertical Flip", vertical_img )
    cv2.imshow("Both", both_img)
    cv2.imshow("orginal", image)
    cv2.waitKey(0)
    '''

    return dict_of_image_names_with_its_gcp_cordinates


def crop_image(image, coordinates, dict_of_image_names_with_its_gcp_coordinates):
    im = cv2.imread(image)
    coordinates = json.loads(coordinates)
    name = str(image).split("\\")[-1]

    height, width, _ = im.shape
    area = 500

    x1 = 0
    x2 = area

    add_a_false_positive = True
    net_data = width/area
    net_height = height / area

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

            crop = im[ py1 - y_pad:py2, x1-x_pad:x2]
            new_cord = check_if_coordinate_in_crop(x1-x_pad, py1 - y_pad, x2, py2, coordinates)
            if len(new_cord) > 0:
                filename = os.path.join(folder_name, name + "_" + str(i) + "_"+ str(h) + '.jpg')
                cv2.imwrite(filename, crop)
                dict_of_image_names_with_its_gcp_coordinates[filename] = new_cord
                dict_of_image_names_with_its_gcp_coordinates = \
                    augment_image(crop, dict_of_image_names_with_its_gcp_coordinates, new_cord, name + "_" + str(i)
                                  + "_" + str(h))
            else:
                if add_a_false_positive is True:
                    add_a_false_positive = False
                    filename = os.path.join(folder_name, name + "_" + str(i) + "_" + str(h) + '.jpg')
                    cv2.imwrite(filename, crop)
                    dict_of_image_names_with_its_gcp_coordinates[filename] = [[0, 0]]

            py1 = py1 + area
            py2 = py2 + area
        x1 = x1 + area
        x2 = x2 + area

    return dict_of_image_names_with_its_gcp_coordinates


if __name__ == "__main__":
    path_to_image = r'C:\Users\dj\Downloads\athul_workspace\NH-150_Flight\DJI_0036.JPG'
    crop_image(path_to_image , '[[1728.1438983674307, 2599.654737926621]]')
