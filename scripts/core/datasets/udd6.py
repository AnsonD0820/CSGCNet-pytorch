#!/usr/bin/python
# -*- encoding: utf-8 -*-


import json
# import cv2
import os

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from ..datasets.transform import *
from torch.utils.data import Dataset


#crop size
# img_w = 769
# img_h = 769
#
# def rand_crop(data, label):
#     width1 = random.randint(0, data.size[0] - img_w)
#     height1 = random.randint(0, data.size[1] - img_h)
#     width2 = width1 + img_w
#     height2 = height1 + img_h
#
#     data = data.crop((width1, height1, width2, height2))
#     label = label.crop((width1, height1, width2, height2))
#
#     return data, label

class UDD6(Dataset):
    def __init__(self, params, mode='train'):
        super(UDD6, self).__init__()
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

        for img, lal in zip(imgs,lals):
        # for img in imgs :
            #img_id = img.split(".png")[0]
            img_path = self.img_dir + img
            #print(img_id)
            lb_id = lal.split('.')[0]
            # lb_id = img.split('.')[0]
            lb_path = self.lb_dir +lb_id+".png"
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
        lb_pth = example["label_img_path"]
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

    # def convert_labels(self, label):
    #     for k, v in self.lb_map.items():
    #         label[label == k] = v
    #     return label
    #
    #     self.ignore_lb = params["dataset_config"]["ignore_idx"]
    #     self.rootpth = params["dataset_config"]["dataset_path"]
    #     self.cropsize = tuple(params["dataset_config"]["cropsize"])
    #     try:
    #         assert self.mode in ('train', 'val', 'test')
    #     except AssertionError:
    #         print(f"[INFO]: Specified {self.mode} mode not in [train, val, test]")
    #         raise
    #     try:
    #         assert os.path.exists(self.rootpth)
    #     except AssertionError:
    #         print(f"[INFO]: Specified dataset path {self.rootpth} does not exist!")
    #         raise
    #
    #     with open(self.config_file, 'r') as fr:
    #         labels_info = json.load(fr)
    #     self.lb_map = {el['id']: el['trainId'] for el in labels_info}
    #
    #     """ Parse Image Directory """
    #     self.imgs = {}
    #     imgnames = []
    #     impth = osp.join(self.rootpth, self.mode,'src') #   cd到train文件夹
    #     folders = os.listdir(impth)
    #     for fd in folders:
    #         img_th = osp.join(impth, fd)
    #         img_name = fd
    #         print(img_th)
    #         print(img_name)
    #         imgnames.extend(img_name)
    #         self.imgs.update(dict(zip(img_name, img_th)))
    #
    #     """ Parse GT Directory """
    #     self.labels = {}
    #     gtnames = []
    #     gtpth = osp.join(self.rootpth,  self.mode,'gt')
    #     folders = os.listdir(gtpth)
    #     for fd in folders:
    #         lb_th = osp.join(gtpth, fd)
    #         lb_name = fd
    #         gtnames.extend(img_name)
    #         print(lb_th)
    #         print(lb_name)
    #         self.labels.update(dict(zip(lb_name, lb_th)))
    #
    #     self.imnames = imgnames
    #     self.len = len(self.imnames)
    #     # print(img_name)
    #     # print(gtnames)
    #     assert set(imgnames) == set(gtnames)
    #     assert set(self.imnames) == set(self.imgs.keys())
    #     assert set(self.imnames) == set(self.labels.keys())
    #
    #     """ Pre-processing and Data Augmentation """
    #     self.to_tensor = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #         ])
    #     self.trans_train = Compose([
    #         ColorJitter(
    #             brightness = 0.5,
    #             contrast = 0.5,
    #             saturation = 0.5),
    #         HorizontalFlip(),
    #         RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
    #         RandomCrop(self.cropsize)
    #         ])
    #
    #
    # def __getitem__(self, idx):
    #     print(idx)
    #     fn  = self.imnames[idx]
    #     impth = self.imgs[fn]
    #     lbpth = self.labels[fn]
    #     img = Image.open(impth)
    #     label = Image.open(lbpth)
    #     if self.mode == 'train':
    #         im_lb = dict(im = img, lb = label)
    #         im_lb = self.trans_train(im_lb)
    #         img, label = im_lb['im'], im_lb['lb']
    #     img = self.to_tensor(img)
    #     label = np.array(label).astype(np.int64)[np.newaxis, :]
    #     label = self.convert_labels(label)
    #     return img, label
    #
    #
    # def __len__(self):
    #     return self.len
    #
    #
    # def convert_labels(self, label):
    #     for k, v in self.lb_map.items():
    #         label[label == k] = v
    #     return label
    #
    #

# if __name__ == "__main__":
#     from tqdm import tqdm
#     with open('../../configs/train_udds.json', "r") as f:
#         params = json.loads(f.read())
#     ds = UDD6(params, mode='val')
#     uni = []
#     for im, lb in tqdm(ds):
#         lb_uni = np.unique(lb).tolist()
#         uni.extend(lb_uni)
#     print(set(uni))
