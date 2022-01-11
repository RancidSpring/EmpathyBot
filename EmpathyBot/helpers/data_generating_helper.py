import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm


class GenerateData:
    def __init__(self, data_path):
        """
        Generate_data class
        Two methods to be used
        1-split_test
        2-save_images
        [Note] that you have to split the public and private from fer2013 file
        """
        self.data_path = data_path

    def split_test(self):
        """
        Helper function to split the validation and test data from general test file as
        it contains (Public test, Private test).
            params:-
                data_path = path to the folder that contains the test data file
        """

        # Load the whole dataset
        csv_path = self.data_path + "/" + 'icml_face_data.csv'
        whole_dataset = pd.read_csv(csv_path)

        # Get the training data
        training_data = whole_dataset.loc[whole_dataset[" Usage"] == "Training"]\
            .drop(columns=[" Usage"]).rename(columns={" pixels": "pixels"})

        # Get the validation data
        validation_data = whole_dataset.loc[whole_dataset[" Usage"] != "PublicTest"]\
            .drop(columns=[" Usage"]).rename(columns={" pixels": "pixels"})

        # Get the testing data
        test_data = whole_dataset.loc[whole_dataset[" Usage"] != "PrivateTest"]\
            .drop(columns=[" Usage"]).rename(columns={" pixels": "pixels"})

        training_data.to_csv(self.data_path+"/train.csv")
        validation_data.to_csv(self.data_path+"/val.csv")
        test_data.to_csv(self.data_path+"/test.csv")
        print("Done splitting the whole dataset into train, val and test data")

    def str_to_image(self, str_img=' '):
        """
        Convert string pixels from the csv file into image object
            params:- take an image string
            return :- return PIL image object
        """
        img_array_str = str_img.split(' ')
        img_array = np.asarray(img_array_str, dtype=np.uint8).reshape(48, 48)
        return Image.fromarray(img_array)

    def save_images(self, data_type='train'):
        """
        save_images is a function responsible for saving images from data files e.g(train, test) in a desired folder
            params:-
            data_type= str e.g (train, val, test)
        """
        folder_name = self.data_path+"/"+data_type
        csv_file_path = self.data_path+"/"+data_type+'.csv'
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        data = pd.read_csv(csv_file_path)
        images = data['pixels']
        number_of_images = images.shape[0]
        for index in tqdm(range(number_of_images)):
            img = self.str_to_image(images[index])
            img.save(os.path.join(folder_name, '{}{}.jpg'.format(data_type, index)), 'JPEG')
        print('Done saving {} data'.format(folder_name))
