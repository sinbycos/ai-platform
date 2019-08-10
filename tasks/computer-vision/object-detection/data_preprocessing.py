# -*- coding: utf-8 -*-
"""
Data preprocessing by creating labels and names for a dataset
"""

import os
import csv
import shutil
import random
from tqdm import tqdm
import cv2 as cv
import numpy as np
import os.path
import matplotlib.pyplot as plt
import sys
import mlflow
# %matplotlib inline

def upload():
    uploaded = files.upload() 
    for name, data in uploaded.items():
        with open(name, 'wb') as f:
            f.write(data)
        print ('saved file', name)

def download(path):
    files.download(path)

def img_show(path):
    image = cv.imread(path)
    height, width = image.shape[:2]
    resized_image = cv.resize(image,(3*width, 3*height), interpolation = cv.INTER_CUBIC)

    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    plt.axis("off")
    #plt.rcParams['figure.figsize'] = [10, 5]
    plt.imshow(cv.cvtColor(resized_image, cv.COLOR_BGR2RGB))
    plt.show()


"""## File Paths and variables"""


img_path = '/content/darknet/data/nabirds/images'
bbox_path = '/content/darknet/data/nabirds/bounding_boxes.txt'
label_path = '/content/darknet/data/nabirds/image_class_labels.txt'
class_path = '/content/darknet/data/nabirds/classes.txt'
size_path = '/content/darknet/data/nabirds/sizes.txt'
names_path = '/content/darknet/data/nabirds/nabirds.names'
txt_path = '/content/darknet/data/nabirds/images'
train_txt = '/content/darknet/data/nabirds/birds_train.txt'
test_txt = '/content/darknet/data/nabirds/birds_test.txt'

data_dict = {}
new_labels = {}
old_labels = {}
img_names = []
label_set = set()

"""## Creating the Labels and Text Files Needed for Training

### Helper Functions
"""

def create_labels(idx, label):
    if label not in label_set:
      label_set.add(label)
      new_labels[label] = len(new_labels.keys())

def get_bird_name(line):
    line = line[0].split()
    label = ''
    
    for i in range(1, len(line)):
        if line[i][0] == '(':
            break
        else:
            label += ' ' + line[i]
            
    label = label[1:]
    
    return line[0], label


"""### Mapping Between the Old Labels and the New Labels"""

cur_line = 0
print(class_path)
with open(class_path) as class_file:
    class_reader = csv.reader(class_file, delimiter = ',')
    for line in class_reader:        
        if cur_line < 295:
          pass
        else:
          idx, label = get_bird_name(line)
          create_labels(int(idx), label)
        cur_line +=1

with open(label_path) as labels_file:
    labels_reader = csv.reader(labels_file, delimiter = ',')
    for line in labels_reader:
        line = line[0].split()
        img = line[0].replace('-', '')
        old_label = old_labels[int(line[1])]
        new_label = new_labels[old_label]
        data_dict[img] = [new_label]

"""### Getting the Image Dimensions"""

with open(size_path) as size_file:
    size_reader = csv.reader(size_file, delimiter = ',')
    for line in size_reader:
        line = line[0].split()
        img = line[0].replace('-', '')
        width = float(line[1])
        height = float(line[2])
        data_dict[img].extend((width, height))

"""### Getting the Location of the Bounding Box and Normalizing it by the Image Dimensions"""

with open(bbox_path) as bbox_file:
    bbox_reader = csv.reader(bbox_file, delimiter = ',')
    for line in bbox_reader:
      line = line[0].split()
      img = line[0].replace('-', '')
      x = float(line[1])
      y = float(line[2])
      w = float(line[3])
      h = float(line[4])
      x = (x + w/2) / data_dict[img][1]
      x = (x + w/2) / data_dict[img][1]
      w /= data_dict[img][1]
      h /= data_dict[img][2]
      del data_dict[img][-1]
      del data_dict[img][-1]
      data_dict[img].extend((x, y, w, h))

"""### Checking Everything is Still Okay"""

len(data_dict.keys())

count = 0
for key in data_dict.keys():
    print(key, data_dict[key])
    if count > 5:
        break
    count += 1

"""### Creating .names File Which Contains All Our Class Labels"""

# create nabirds.names
f = open(names_path, 'w')
for key in new_labels.keys():
    f.write(key + '\n')
f.close()

"""### Creating _train.txt and _test.txt Which Contains the Location to the Images and Set Which Images Will Be Used to Train"""

num_imgs = len(data_dict.keys())
test_size = int(0.1 * num_imgs)
test_idx = random.sample(range(0, num_imgs), test_size)
for subdir, dirs, imgs in os.walk(img_path):
    for img in imgs:
        old_path = subdir + '/' + img
        new _path = img_path + '/' + img
        shutil.move(old_path, new_path)
        img_names.append(new_path)
        
        
f_train = open(train_txt, 'w')
f_test = open(test_txt, 'w')
for i, img in tqdm(enumerate(img_names)):
    if i in test_idx:
      f_test.write(img + '\n')
    else:
      f_train.write(img + '\n')
f_train.close()
f_test.close()

"""### Writing the Label .txt File for Each Image"""

# create the labels for each image
for key in tqdm(data_dict.keys()):
    f_name = txt_path + '/' + key + '.txt'
    line = ''
    for col in data_dict[key]:
        line += ' ' + str(col)
    line = line[1:]
    f = open(f_name, 'w')
    f.write(line)
    f.close()


