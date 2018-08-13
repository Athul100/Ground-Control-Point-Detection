import os
from create_dataset import create_dataset
from model import ModelClass
from aug_image import crop_image
from six.moves import cPickle as pickle
import pandas as pd
import os


class InitialiseData:

    def __init__(self, data_set_path, gcp_filename):
        print("Initialising the parameters")
        print(f"Data set path : {data_set_path} , GCP File names : {gcp_filename}")
        self.data_set_path = data_set_path
        self.gcp_filename = gcp_filename

    def get_gcp_dataframe(self):
        return pd.read_excel(self.gcp_filename)

    def prepare(self):
        df = self.get_gcp_dataframe()
        dict_of_image_names_with_its_gcp_cordinates = {}
        pickle_exist = False
        dict_pickle_path = os.path.join('pickle', 'dict.pickle')
        if os.path.exists(dict_pickle_path):
            with open(dict_pickle_path, 'rb') as (f):
                dict_of_image_names_with_its_gcp_cordinates = pickle.load(f)
                pickle_exist = True
        for index, row in enumerate(df.iterrows()):
            if pickle_exist:
                break
            print(index)
            image_name: str = df['FileName'].iloc[index]
            gcp_location: str = df['GCPLocation'].iloc[index]
            print(gcp_location, image_name)
            path = os.path.join(self.data_set_path, image_name)
            if not os.path.exists(path):
                with open('not_found.txt', 'a') as the_file:
                    the_file.write(path + '\n')
                continue
            dict_of_image_names_with_its_gcp_cordinates = crop_image(path, gcp_location,
                                                                     dict_of_image_names_with_its_gcp_cordinates)

            with open(dict_pickle_path, 'wb') as (f):
                pickle.dump(dict_of_image_names_with_its_gcp_cordinates, f, pickle.HIGHEST_PROTOCOL)

        data_set, target_set = create_dataset(dict_of_image_names_with_its_gcp_cordinates)

        print(target_set[0])
        print(data_set.shape)

        print("loading Model")
        model = ModelClass()
        model.create_model()
        model.train_model(data_set, target_set)


if __name__ == "__main__":
    path_to_data_set: str = r'C:\Users\dj\Downloads\athul_workspace'
    gcp_details_filename: str = r'C:\Users\dj\Downloads\athul_workspace\cleaned.xlsx'
    init = InitialiseData(path_to_data_set, gcp_details_filename)
    init.prepare()

