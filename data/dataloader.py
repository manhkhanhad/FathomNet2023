
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
            self.transform = transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.Resize((512, 512)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if is_train:
            data_annotation = pd.read_csv(label_path)
            self.images = data_annotation['id']
            self.label = data_annotation['categories']
        else:
            self.images =[i[:-4] for i in os.listdir(self.image_dir)]
        self.num_classes = 133
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx] + ".png")
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        if self.is_train:
            label = [int(i) for i in self.label[idx][1:-1].split(',')]
            label_tensor = torch.zeros(self.num_classes)
            label_tensor[label] = 1
            return image, label_tensor
        else:
            return image, image_path


class FathomNetLoader(data.Dataset):
    def __init__(self, root='./datasets/train', annFile="./datasets/train.json", transform=None, target_transform=None):
        super().__init__()
        """
        Args:
            - root (string): Root dir where images are downloaded to
            - annFile (string): Path to json annotation file
            - transform (callable, optional): A function/transform that takes  in an PIL image and return a transformed version. E.g, `transforms.ToTensor()`
            - target_transform (callable, optional): takes in the target and transforms it
        """
        from pycocotools.coco import COCO
        self.root = root 
        self.coco = COCO(annFile)
        self.n_classes = 290
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.split = self.root[-4:]
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        if self.split == 'test':
            label = []
        else:
            ann_ids = coco.getAnnIds(imgIds=img_id)
            target = coco.loadAnns(ann_ids)[0]['category_id']
            target = int(target)
            
            label = torch.zeros(self.n_classes)
            label[target] = 1

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.split == 'eval':
            return img, path
        else:
            return img, label

    def __len__(self):
        return len(self.ids)
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str