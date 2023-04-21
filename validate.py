import numpy as np
import torch
from sklearn import metrics
import torchvision

def validate(args, model, val_loader, device):
    model.eval()
    n_classes = val_loader.dataset.num_classes
    gts = {i:[] for i in range(0, n_classes)}
    preds = {i:[] for i in range(0, n_classes)}
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            pred = outputs.squeeze().data.cpu().numpy()
            gt = labels.squeeze().data.cpu().numpy()

            for label in range(0, n_classes):
                gts[label].extend(gt[:,label])
                preds[label].extend(pred[:,label])

    FinalMAPs = []
    for i in range(0, n_classes):
        precision, recall, thresholds = metrics.precision_recall_curve(gts[i], preds[i])
        FinalMAPs.append(metrics.auc(recall, precision))
    # print(FinalMAPs)

    return np.mean(FinalMAPs)