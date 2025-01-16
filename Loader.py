from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as data
from os import listdir
from os.path import join
import numpy as np
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path):
    return Image.open(path).convert('RGB')

class Dataset(data.Dataset):
    def __init__(self,contentPath,stylePath,fineSize):
        super(Dataset,self).__init__()
        self.contentPath = contentPath
        self.image_list = [x for x in listdir(contentPath) if is_image_file(x)]
        self.stylePath = stylePath
        self.fineSize = fineSize
        #self.normalize = transforms.Normalize(mean=[103.939,116.779,123.68],std=[1, 1, 1])
        #normalize = transforms.Normalize(mean=[123.68,103.939,116.779],std=[1, 1, 1])
        self.prep = transforms.Compose([
                    transforms.Resize(fineSize),
                    transforms.ToTensor(),
                    #transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                    ])

    def __getitem__(self,index):
        contentImgPath = os.path.join(self.contentPath,self.image_list[index])
        styleImgPath = os.path.join(self.stylePath,self.image_list[index])
        contentImg = default_loader(contentImgPath)
        styleImg = default_loader(styleImgPath)

        # resize
        if(self.fineSize != 0):
            w,h = contentImg.size
            if(w > h):
                if(w != self.fineSize):
                    neww = self.fineSize
                    newh = int(h*neww/w)
                    contentImg = contentImg.resize((neww,newh))
                    styleImg = styleImg.resize((neww,newh))
            else:
                if(h != self.fineSize):
                    newh = self.fineSize
                    neww = int(w*newh/h)
                    contentImg = contentImg.resize((neww,newh))
                    styleImg = styleImg.resize((neww,newh))


        # Preprocess Images
        contentImg = transforms.ToTensor()(contentImg)
        styleImg = transforms.ToTensor()(styleImg)
        return contentImg.squeeze(0),styleImg.squeeze(0),self.image_list[index]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)

class GaussianDataset(Dataset):
    def __init__(self, n1, n2, d):
        """
        Args:
            n (int): Number of samples in the dataset.
            d (int): Dimensionality of each sample.
        """
        super().__init__()
        # Pre-generate all samples from N(0, I)
        self.data = torch.randn(size=(n1,n2,d))  # shape: [n, d]
        self.n1 = n1
        self.n2 = n2

    def __len__(self):
        return self.n1*self.n2

    # def __getitem__(self, idx):
    #     """
    #     Return the idx-th sample.
    #     """
    # return self.data[idx]

    def __getitem__(self, idx):
        # Convert the flattened index back to (i1, i2)
        i1 = idx // self.n2
        i2 = idx % self.n2

        # Get the sample corresponding to indices (i1, i2)
        sample = self.data[:, :, i1, i2]  # Shape: (k, d)

        # Return both the sample and its corresponding indices (i1, i2)
        return sample, (i1, i2)
