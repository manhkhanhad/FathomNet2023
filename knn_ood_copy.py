import torch
import faiss
import numpy as np
import os
import json
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'
feature_dir = "/storageStudents/danhnt/khanhngo/FathomNet2023/feature_1/"

normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
prepos_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(448, 960)]))# Last Layer only

#Load InDomain feature
ID_feature = np.load(os.path.join(feature_dir, "InDomain_Resnet101_feature.npy"))
ID_feature = prepos_feat(ID_feature)
ID_score = np.load(os.path.join(feature_dir, "InDomain_Resnet101_score.npy"))
# ID_label = np.load(os.path.join(feature_dir, "InDomain_resnet18_label.npy"))
#Load OutDomain feature
OD_feature = np.load(os.path.join(feature_dir, "OutDomain_Resnet101_feature.npy"))
OD_feature = prepos_feat(OD_feature)
OD_score = np.load(os.path.join(feature_dir, "OutDomain_Resnet101_score.npy"))
OD_score = torch.sigmoid(torch.tensor(OD_score)).data.cpu().numpy()
#################### KNN score OOD detection #################
label_threshold = 0.4

index = faiss.IndexFlatL2(ID_feature.shape[1])
index.add(ID_feature)

label_mapping = json.load(open(os.path.join("mapping_label.json"), "r"))
# Calculate lamda threshold
id_score = []

for i in range(ID_feature.shape[0]):
    D, id = index.search(ID_feature[i].reshape(1, -1), 50)
    ood_score = np.mean(np.linalg.norm(np.tile(OD_feature[i], (50, 1)) - ID_feature[id[0,:], :], ord = 2, axis = 1)[1:])
    id_score.append(ood_score)

sorted_lst = np.sort(id_score, )
idx = int(0.05 * len(sorted_lst))
lambda_ = sorted_lst[idx]


image_id = []
pred_labels = []
osd = []
with open('OOD_image_path.txt', 'r') as file:
    for line in file:
        image_id.append(line.strip())
        
for i in range(OD_feature.shape[0]):
    score = OD_score[i]
    D, id = index.search(OD_feature[i].reshape(1, -1), 50)
    ood_score = np.mean(np.linalg.norm(np.tile(OD_feature[i], (50, 1)) - ID_feature[id[0,:], :], ord = 2, axis = 1))
    if ood_score > lambda_:
        is_ood = 1
    else:
        is_ood = 0
    
    #label
    labels = torch.ones(score.shape).cuda() * (score >= 0.001)
    # Loop over the predictions and extract the indices where the value is 1
    for j in range(labels.shape[0]):
        indices = torch.where(labels[j] == 1)[0].tolist()
        categories = [str(index) for index in indices]
        if not len(categories):
            categories = ['1', '2']
    
    
    
#     pred_label = " ".join(str(x) for x in pred_label)
#     pred_labels.append(pred_label)
#     osd.append(is_ood)
    print("Image: ", image_id[i], " OOD: ", is_ood, " Label: ", pred_label)
    
# result_pd = pd.DataFrame({"id": image_id, "categories": pred_labels, "osd": osd})
# result_pd.to_csv("OOD_result.csv", index = False)