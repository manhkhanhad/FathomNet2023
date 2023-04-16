
from torch.utils import data
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import os
from PIL import Image

class fathomnetDataLoader(data.Dataset):
    def __init__(self, root_image, label_path, is_train = True):
        super(fathomnetDataLoader, self).__init__()
        self.image_dir = root_image + "/train" if is_train else root_image + "/test"
        self.label_path = label_path
        self.is_train = is_train
        if is_train:
            self.transform =  transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            torchvision.transforms.Resize((512, 512)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             ])
        else:
            transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.Resize((512, 512)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        data_annotation = pd.read_csv(label_path)
        self.images = data_annotation['id']
        self.label = data_annotation['categories']
        self.num_classes = 133
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx] + ".png")
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = [int(i) for i in self.label[idx][1:-1].split(',')]
        label_tensor = torch.zeros(self.num_classes)
        label_tensor[label] = 1
        return image, label_tensor
