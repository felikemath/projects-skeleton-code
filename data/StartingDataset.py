import torch
from torchvision import transforms
from torchvision import datasets
import pandas as pd
from PIL import Image


class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self):
        self.data = pd.read_csv("train.csv")



    def __getitem__(self, index):
        inputs = torch.zeros([3, 224, 224])
        label = self.data.iloc[index][1]
        convert_tensor = transforms.ToTensor()
        img = Image.open("train\{0}".format(self.data.iloc[index][0]))
        inputs = convert_tensor(img)
        # img.show() to display image


        return inputs, label

    def __len__(self):
        return len(self.data)
