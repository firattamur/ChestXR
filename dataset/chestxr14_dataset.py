import os
import torch
import pandas as pd
import imageio.v3 as iio
from torchvision import transforms
from torch.utils.data import Dataset


class ChestXRDataset(Dataset):

    def __init__(self, args):
        """

        Custom dataset for ChestXR-14 dataset.

        :param args: input arguments parameters

        """

        self.root      = args.root
        self.is_train  = args.is_train
        self.extension = args.extension

        self.path_images      = os.path.join(self.root, "images")
        self.path_labels_csv  = os.path.join(self.root, "csvs", "labels_encoded.csv")
        self.labels_encoded   = pd.read_csv(self.path_labels_csv)

        if self.is_train:

            self.path_txt = os.path.join(self.root, "txts", "train_val_list.txt")
            self.transforms = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                ]
            )

        else:

            self.path_txt = os.path.join(self.root, "txts", "test.txt")
            self.transforms = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                ]
            )

        self.image_paths: list = []

        with open(self.path_txt, 'r') as file:

            for name in file.readlines():
                path_image = os.path.join(self.path_images, name.strip())

                if os.path.exists(path_image):
                    self.image_paths.append(path_image)

    def __len__(self) -> int:
        """

        Return length of dataset.

        :return:
        """
        return len(self.image_paths)

    def __getitem__(self, item: int) -> tuple:
        """

        Return a single item from dataset.

        :param item : index of the item
        :return     : a pair of image and label
            :shape: (np.ndarray)

        """
        image_path = self.image_paths[item]
        image_name = image_path.split("/")[-1]

        label = self.labels_encoded[self.labels_encoded["Image Index"] == image_name].drop(["Image Index"],
                                                                                           axis=1).values[0]
        image = iio.imread(image_path)
        image = torch.tensor(image).unsqueeze(0)

        return self.transforms(image), torch.tensor(label)





