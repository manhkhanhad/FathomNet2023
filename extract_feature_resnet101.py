
import argparse
import torch
import torch.nn as nn
from data.dataloader import fathomnetDataLoader, FathomNetLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch.utils.data as data

from models.simple_classifier import *
import torchvision
from torchvision import transforms

def coco_collate(batch):
    # Define the transformations to be applied to each image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images = []
    targets = []

    for sample in batch:
        images.append(transform(sample[0]))
        targets.append(sample[1])

    images = torch.stack(images, dim=0)

    return images, targets

def extract_InDomain_feature():
    
    train_dataset = FathomNetLoader(root = "/storageStudents/danhnt/khanhngo/dataset/fathomnet/train", annFile="/storageStudents/danhnt/khanhngo/dataset/object_detection/train.json")
    dataloader = data.DataLoader(train_dataset, batch_size=64, num_workers=1, shuffle=False, pin_memory=True, collate_fn=coco_collate)
    
    #Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_dict = torch.load('/storageStudents/danhnt/khanhngo/FathomNet2023/checkpoint/resnet101_best.pth')
    clsfier_dict = torch.load('/storageStudents/danhnt/khanhngo/FathomNet2023/checkpoint/resnet101clsfier_best.pth')
    orig_resnet = torchvision.models.resnet101(pretrained=True)
    features = list(orig_resnet.children())

    model= nn.Sequential(*features[0:8])
    clsfier = SimpleClassifier(2048, 290)
    model.load_state_dict(model_dict)
    clsfier.load_state_dict(clsfier_dict)
    
    model.cuda()
    clsfier.cuda()
    
    cache_name = "/storageStudents/danhnt/khanhngo/FathomNet2023/feature_1/InDomain_Resnet101_"
    
    feat_log = []
    score_log = []
    label_log = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.cuda()
            feature_map = model(inputs)
            preds = torch.sigmoid(feature_map).cuda()
            labels = torch.ones(preds.shape).cuda() * (preds >= 0.001)

            out = F.adaptive_avg_pool2d(feature_map, 1).squeeze()
            # out = torch.cat([F.adaptive_avg_pool2d(layer_feat, 1).squeeze() for layer_feat in feature_list], dim=1)
            
            feat_log.append(out)
            # label_log.append(labels)
            score_log.append(preds)
        
        feat_log = torch.concat(feat_log, 0).data.cpu().numpy()
        # label_log = torch.concat(label_log, 0).data.cpu().numpy()
        score_log = torch.concat(score_log, 0).data.cpu().numpy()
        
        np.save(cache_name + "feature.npy", feat_log)
        # np.save(cache_name + "label.npy", label_log)
        np.save(cache_name + "score.npy", score_log)
        
def extract_OutDomain_feature():
    train_dataset = FathomNetLoader(root = "/storageStudents/danhnt/khanhngo/dataset/fathomnet/test", annFile="/storageStudents/danhnt/khanhngo/dataset/object_detection/eval.json")
    dataloader = data.DataLoader(train_dataset, batch_size=64, num_workers=1, shuffle=False, pin_memory=True, collate_fn=coco_collate)
    
    #Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_dict = torch.load('/storageStudents/danhnt/khanhngo/FathomNet2023/checkpoint/resnet101_best.pth')
    clsfier_dict = torch.load('/storageStudents/danhnt/khanhngo/FathomNet2023/checkpoint/resnet101clsfier_best.pth')
    orig_resnet = torchvision.models.resnet101(pretrained=True)
    features = list(orig_resnet.children())

    model= nn.Sequential(*features[0:8])
    clsfier = SimpleClassifier(2048, 290)
    model.load_state_dict(model_dict)
    clsfier.load_state_dict(clsfier_dict)
    
    model.cuda()
    clsfier.cuda()
    cache_name = "/storageStudents/danhnt/khanhngo/FathomNet2023/feature_1/OutDomain_Resnet101_"
    image_path = []
    feat_log = []
    score_log = []
    label_log = []
    model.eval()
    with torch.no_grad():
        for inputs, path in tqdm(dataloader):
            
            inputs = inputs.cuda()
            feature_map = model(inputs)
            preds = torch.sigmoid(feature_map).cuda()
            labels = torch.ones(preds.shape).cuda() * (preds >= 0.001)
            image_path += path
            out = F.adaptive_avg_pool2d(feature_map, 1).squeeze()
            # out = torch.cat([F.adaptive_avg_pool2d(layer_feat, 1).squeeze() for layer_feat in feature_list], dim=1)
            
            feat_log.append(out.cpu())
            # label_log.append(labels)
            score_log.append(preds.cpu())
        
        breakpoint()
        feat_log = torch.concat(feat_log, 0).data.numpy()
        score_log = torch.concat(score_log, 0).data.numpy()
        
        np.save(cache_name + "feature.npy", feat_log)
        np.save(cache_name + "score.npy", score_log)
    
    with open('OOD_image_path.txt', 'w') as file:
        for path in image_path:
            file.write(path + '\n')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', type=str, default='resnet18',
                        help='Architecture to use resnet18|resnet101')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch Size')
     #Data
    parser.add_argument('--root_image', type=str, default='/storageStudents/danhnt/khanhngo/dataset/fathomnet')
    parser.add_argument('--label_path', type=str, default='/storageStudents/danhnt/khanhngo/dataset/multilabel_classification/train_refine.csv')
    parser.add_argument('--load_dir', type=str, default="/storageStudents/danhnt/khanhngo/FathomNet2023/checkpoint/best_model.pth",
                        help='Path to load models')
    args = parser.parse_args()
    # extract_InDomain_feature()
    extract_OutDomain_feature()