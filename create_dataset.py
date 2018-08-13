import numpy as np, cv2
from process import process_image


def preprocess_input(x):
    x = x / 255.0
    x = x - 0.5
    x = x * 2.0
    return x


def create_dataset(dict_of_image_names_with_its_gcp_cordinates):

    data_image_height = 500
    data_image_width = 500
    data_image_depth = 3
    max_num_of_gcp = 3

    print(len(dict_of_image_names_with_its_gcp_cordinates))
    is_real_data_required_in_data_set = False
    length_of_dataset = len(dict_of_image_names_with_its_gcp_cordinates)
    if is_real_data_required_in_data_set is True:
        length_of_dataset = length_of_dataset * 2
    if data_image_depth is None:
        data_set = np.ndarray(shape=(length_of_dataset, data_image_height, data_image_width), dtype=np.float32)
    else:
        data_set = np.ndarray(shape=(length_of_dataset, data_image_height, data_image_width,
                                     data_image_depth), dtype=np.float32)

    target_set = np.ndarray(shape=(length_of_dataset, 2 * max_num_of_gcp), dtype=np.float32)

    for index, image_file in enumerate(dict_of_image_names_with_its_gcp_cordinates):
        image_coordinate = dict_of_image_names_with_its_gcp_cordinates[image_file]
        image_data = cv2.imread(image_file)
        h, w, _ = image_data.shape
        print(image_data.shape)
        processed_image = process_image(image_data, image_file)
        processed_image = preprocess_input(processed_image)
        print(processed_image.shape)
        data_set[index, :, :] = processed_image
        if is_real_data_required_in_data_set is True:
            gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            image_data = preprocess_input(gray)
            data_set[index + len(dict_of_image_names_with_its_gcp_cordinates), :, :, :] = image_data
        coordinates = []
        for coordinate in image_coordinate:
            coordinates.append([round((coordinate[0] / w)* data_image_width) , round((coordinate[1] / h) *
                                                                                     data_image_height)])

        while len(coordinates) < max_num_of_gcp:
            coordinates.append([0, 0])

        coordinates = sum(coordinates, [])
        target_set[index, :] = coordinates
        if is_real_data_required_in_data_set is True:
            target_set[index + len(dict_of_image_names_with_its_gcp_cordinates), :] = coordinates

    data_set = data_set[0:length_of_dataset, :, :]
    target_set = target_set[0:length_of_dataset, :]
    print(target_set.shape)
    print(data_set.shape)
    return data_set, target_set


