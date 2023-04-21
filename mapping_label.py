import pandas as pd
import json

raw_label = pd.read_csv("/storageStudents/danhnt/khanhngo/dataset/multilabel_classification/train.csv")
refine_label = pd.read_csv("/storageStudents/danhnt/khanhngo/dataset/multilabel_classification/train_refine.csv")

raw = dict(zip(raw_label["id"], raw_label["categories"]))
refine = dict(zip(refine_label["id"], refine_label["categories"]))


mapping = {}
for id in raw:
    raw_l = raw[id][1:-1].split(",")
    refine_l = refine[id][1:-1].split(",")
    
    for i, j in zip(refine_l, raw_l):
        if int(i) not in mapping:
            mapping[int(i)] = j

with open("mapping_label.json", "w") as f:
    json.dump(mapping, f)