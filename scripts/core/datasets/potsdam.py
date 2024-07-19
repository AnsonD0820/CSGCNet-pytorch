#!/usr/bin/python
# -*- encoding: utf-8 -*-


import json
# import cv2
import os

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from ..datasets.transform import *
# from scripts.core.datasets.transform import *
from torch.utils.data import Dataset


#crop size
# img_w = 769
# img_h = 769
#
# def rand_crop(data, label):/dataset/Potsdam
#     width1 = random.randint(0, data.size[0] - img_w)
#     height1 = random.randint(0, data.size[1] - img_h)
#     width2 = width1 + img_w
#     height2 = height1 + img_h
#
#     data = data.crop((width1, height1, width2, height2))
#     label = label.crop((width1, height1, width2, height2))
#
#     return data, label

class potsdam(Dataset):
    def __init__(self, params, mode='train'):
        super(potsdam, self).__init__()
        self.mode = mode
        self.config_file = params["dataset_config"]["dataset_config_file"]
        self.ignore_lb = params["dataset_config"]["ignore_idx"]
        self.rootpth = params["dataset_config"]["dataset_path"]
        self.cropsize = tuple(params["dataset_config"]["cropsize"])
        self.img_dir = self.rootpth +"/"+ self.mode +"/img/"
        self.lb_dir = self.rootpth + "/"+self.mode +"/lb/"
        imgs = os.listdir(self.img_dir)
        imgs.sort()
        lals = os.listdir(self.lb_dir)
        lals.sort()
        self.examples = []

        for img,lal in zip(imgs,lals) :
        # for img in imgs :
            #img_id = img.split(".png")[0]
            img_path = self.img_dir + img
            #print(img_id)
            lb_id = lal.split('.')[0]
            # lb_id = img.split('.')[0]
            # print(lb_id)
            lb_path = self.lb_dir +lb_id+".png"
            # lb_path = self.lb_dir +lb_id+".jpg"
            example = {}
            example["img_path"] = img_path
            example["label_img_path"] = lb_path
            #example["img_id"] = img_id
            self.examples.append(example)
        self.num_examples = len(self.examples)

        """ Pre-processing and Data Augmentation """
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness = 0.5,
                contrast = 0.5,
                saturation = 0.5),
            HorizontalFlip(),
            RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomCrop(self.cropsize)
            ])

    def __getitem__(self, index):
        example = self.examples[index]
        img_path = example["img_path"]
        #print(img_path)
        img = Image.open(img_path)
        #print(img_path)
        lb_pth =  example["label_img_path"]
        #print(lb_pth)
        label = Image.open(lb_pth)
        #img,label = rand_crop(img,label)
        # if self.mode == "train":
        #    im_lb = dict(im=img, lb=label)
        #    im_lb = self.trans_train(img)
        #    img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        #print(label.shape)
        #label = self.convert_labels(label)
        return img, label

    def __len__(self):
        return self.num_examples
