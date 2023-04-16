import json
import collections
import matplotlib.pyplot as plt

# with open('/storageStudents/danhnt/khanhngo/dataset/object_detection/train.json', 'r') as f:
#     data = json.load(f)

# exist_categories = []
# print("NUM BBOX: ", len(data['annotations']))
# for anot in data['annotations']:
#     label = anot['category_id']
#     exist_categories.append(label)

# # exist_categories = list(exist_categories)

# count_dict = collections.Counter(exist_categories)
# elements = list(count_dict.keys())
# frequencies = list(count_dict.values())

# plt.bar(elements, frequencies)
# plt.xlabel('Elements')
# plt.ylabel('Frequency')
# plt.title('Frequency of Elements in the List')
# plt.savefig('Train_class_distribution.png')

# exist_categories = list(set(exist_categories))
# refind_categories = []
# for item in data['categories']:
#     if item['id'] in exist_categories:
#         refind_categories.append(item)
# print(len(refind_categories))

# refine_label = data
# refine_label['categories'] = refind_categories

# with open('/storageStudents/danhnt/khanhngo/dataset/object_detection/train_refine.json', 'w') as f:
#     json.dump(refine_label, f)


import pandas as pd
data = pd.read_csv('/storageStudents/danhnt/khanhngo/dataset/multilabel_classification/train.csv')
label = data['categories'].tolist()

categories_mapping = {}
new_label = []
for i in label:
    for j in i[1:-1].split(','):
        if int(float(j)) not in categories_mapping:
            categories_mapping[int(float(j))] = len(categories_mapping)
    new_label.append([categories_mapping[int(float(k))] for k in i[1:-1].split(',')])

print(len(categories_mapping))
new_data = pd.DataFrame({'id': data['id'], 'categories': new_label})
new_data.to_csv('/storageStudents/danhnt/khanhngo/dataset/multilabel_classification/train_refine.csv', index=False)