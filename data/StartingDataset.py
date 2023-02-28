import torch
from torchvision import transforms
from torchvision import datasets
import pandas as pd
from PIL import Image


class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    Number of data: 25361
    Number of classes: 5005
    """

    def __init__(self):
        self.data = pd.read_csv("C://Users//MichaelSong//Documents//GitHub//projects-skeleton-code//data//train.csv")
        self.label_dict = dict()
        cc = 1
        for i in range(len(self.data)):
            if self.data.iloc[i][1] not in self.label_dict:
                self.label_dict[self.data.iloc[i][1]] = cc
                cc += 1



    def __getitem__(self, index):
        inputs = torch.zeros([3, 224, 224])
        label = self.data.iloc[index][1]

        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        img = Image.open("C://Users//MichaelSong//Documents//GitHub//projects-skeleton-code//data//train//{0}".format(self.data.iloc[index][0]))

        inputs = train_transform(img)
        inputs.resize_(3, 224, 224)
        # img.show() to display image


        return inputs, label

    def __len__(self):
        return len(self.data)
