import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


class PlainDataset(Dataset):
    def __init__(self, csv_file, img_dir, data_type, transform):
        """
        PyTorch Dataset class
        :param csv_file: the path of the csv file    (train, validation, test)
        :param img_dir: the directory of the images (train, validation, test)
        :param data_type: string for searching along the image_dir (train, val, test)
        :param transform: PyTorch transformation over the data
        """

        self.csv_file = pd.read_csv(csv_file)
        self.labels = self.csv_file['emotion']
        self.img_dir = img_dir
        self.transform = transform
        self.data_type = data_type

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = Image.open(self.img_dir + self.data_type + str(idx) + '.jpg')
        labels = np.array(self.labels[idx])
        labels = torch.from_numpy(labels).long()

        if self.transform:
            img = self.transform(img)
        return img, labels


# Helper function
def eval_data_data_loader(csv_file, img_dir, data_type, sample_number, transform=None):
    """
    Helper function used to evaluate the Dataset class
    :param csv_file: the path of the csv file    (train, validation, test)
    :param img_dir: the directory of the images (train, validation, test)
    :param data_type: string for searching along the image_dir (train, val, test)
    :param sample_number: any number from the data to be shown
    :param transform: the transformation of an image
    :return:
    """
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = PlainDataset(csv_file=csv_file, img_dir=img_dir, data_type=data_type, transform=transform)

    label = dataset.__getitem__(sample_number)[1]
    print(label)
    imgg = dataset.__getitem__(sample_number)[0]
    imgnumpy = imgg.numpy()
    imgt = imgnumpy.squeeze()
    plt.imshow(imgt)
    plt.show()
