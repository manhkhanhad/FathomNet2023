from models.resnet_supcon import SupConResNet
import argparse
import torch
import torch.nn as nn
from data.dataloader import fathomnetDataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch.utils.data as data

def extract_InDomain_feature():
    train_dataset = fathomnetDataLoader(root_image = args.root_image, label_path = args.label_path, is_train = True)
    num_classes = train_dataset.num_classes
    dataloader = data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 0)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SupConResNet(name = args.arch, head='mlp', num_classes = num_classes).to(device)
    print("Load model from ", args.load_dir)
    cache_name = "/storageStudents/danhnt/khanhngo/FathomNet2023/feature/InDomain_{}_".format(args.arch)
    model.load_state_dict(torch.load(args.load_dir))
    
    feat_log = []
    score_log = []
    label_log = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            
            score, feature_list = model.feature_list(inputs)
            out = torch.cat([F.adaptive_avg_pool2d(layer_feat, 1).squeeze() for layer_feat in feature_list], dim=1)
            
            feat_log.append(out)
            label_log.append(labels)
            score_log.append(score)
        
        feat_log = torch.concat(feat_log, 0).data.cpu().numpy()
        label_log = torch.concat(label_log, 0).data.cpu().numpy()
        score_log = torch.concat(score_log, 0).data.cpu().numpy()
        
        np.save(cache_name + "feature.npy", feat_log)
        np.save(cache_name + "label.npy", label_log)
        np.save(cache_name + "score.npy", score_log)
        
def extract_OutDomain_feature():
    train_dataset = fathomnetDataLoader(root_image = args.root_image, label_path = args.label_path, is_train = False)
    num_classes = train_dataset.num_classes
    dataloader = data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 0)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SupConResNet(name = args.arch, head='mlp', num_classes = num_classes).to(device)
    print("Load model from ", args.load_dir)
    cache_name = "/storageStudents/danhnt/khanhngo/FathomNet2023/feature/OutDomain_{}_".format(args.arch)
    model.load_state_dict(torch.load(args.load_dir))
    image_path = []
    feat_log = []
    score_log = []
    label_log = []
    model.eval()
    with torch.no_grad():
        for inputs, path in tqdm(dataloader):
            inputs = inputs.to(device)
            image_path += path
            score, feature_list = model.feature_list(inputs)
            out = torch.cat([F.adaptive_avg_pool2d(layer_feat, 1).squeeze() for layer_feat in feature_list], dim=1)
            
            feat_log.append(out)
            score_log.append(score)
        
        feat_log = torch.concat(feat_log, 0).data.cpu().numpy()
        score_log = torch.concat(score_log, 0).data.cpu().numpy()
        
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