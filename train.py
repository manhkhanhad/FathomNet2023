import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data.dataloader import fathomnetDataLoader
from models.resnet_supcon import SupConResNet
from validate import validate
import os
from tqdm import tqdm

def Train():
    # Define transformations for the dataset
    train_dataset = fathomnetDataLoader(root_image = args.root_image, label_path = args.label_path, is_train = True)
    num_classes = train_dataset.num_classes
    dataloader = data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 0)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SupConResNet(name = args.arch, head='mlp', num_classes = num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr = args.l_rate)
    bceloss = nn.BCEWithLogitsLoss()
    print("USING: ", device)
    best_mAP = 0
    if args.load:
        print("Load model from ", args.load_dir)
        model.load_state_dict(torch.load(args.load_dir))
    
    for epoch in range(args.n_epoch):
        for i, (images, labels) in tqdm(enumerate(dataloader)):
            images = images.to(device)
            labels = labels.to(device)
            
            pred = model(images)
            loss = bceloss(pred, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % args.print_step == 0:
                print('Epoch: {}- Iter:{}/{}, Loss: {}'.format(epoch, i, len(train_dataset), loss.item()))
            
        mAP = validate(args, model, dataloader, device)
        print(mAP)
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'epoch_{}.pth'.format(epoch)))
        if mAP > best_mAP:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', type=str, default='resnet18',
                        help='Architecture to use resnet18|resnet101')
    parser.add_argument('--n_epoch', type=int, default=50,
                        help='# of the epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch Size')

    parser.add_argument('--print_step', type=int, default=10,)
    
    # batch_size 320 for resenet101
    parser.add_argument('--l_rate', type=float, default=1e-4,
                        help='Learning Rate')
    
    #Data
    parser.add_argument('--root_image', type=str, default='/storageStudents/danhnt/khanhngo/dataset/fathomnet')
    parser.add_argument('--label_path', type=str, default='/storageStudents/danhnt/khanhngo/dataset/multilabel_classification/train_refine.csv')

    #save and load
    parser.add_argument('--load', action='store_true', help='Whether to load models')
    parser.add_argument('--save_dir', type=str, default="/storageStudents/danhnt/khanhngo/FathomNet2023/checkpoint",
                        help='Path to save models')
    parser.add_argument('--load_dir', type=str, default="/storageStudents/danhnt/khanhngo/FathomNet2023/checkpoint/best_model.pth",
                        help='Path to load models')
    args = parser.parse_args()
    Train()