import cv2
from core.datasets.uavid import UAVid
from core.datasets.udd6 import UDD6
from core.datasets.potsdam import potsdam
from core.utils.loss import OhemCELoss
from core.utils.optimizer import Optimizer
from evaluate import MscEval
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
from core.models.CSGCNet import CSGCNet
from misc import iou, f1, acc, get_color_pallete_isprs, get_color_pallete_udd6
import torch.optim as optim
from pathlib import Path
import logging
import time
import datetime
import argparse
import numpy as np
import json
from shutil import copyfile
import os

img_path = "/home/wuyi/AnsonD/CSGCNet/Output_RGB_UDD6_CSGCNet"
if not os.path.exists(img_path):
    os.makedirs(img_path)


def test(config):
    with open(config, "r") as f:
        params = json.loads(f.read())

    """ Set Dataset Params """
    n_classes = params["dataset_config"]["num_classes"]
    n_img_per_gpu = params["validation_config"]["batch_size"]
    n_workers = params["training_config"]["num_workers"]
    cropsize = params["dataset_config"]["cropsize"]

    """ Prepare DataLoader """
    if params["dataset_config"]["name"] == "uavid":
        ds_train = UAVid(params, mode='train')
        ds_val = UAVid(params, mode='val')
    elif params["dataset_config"]["name"] == "UDD":
        ds_train = UDD6(params, mode='train')
        ds_val = UDD6(params, mode='val')
        ds_test = UDD6(params, mode='test')
    elif params["dataset_config"]["name"] == "potsdam":
        ds_train = potsdam(params, mode='train')
        ds_val = potsdam(params, mode='val')
        ds_test = potsdam(params, mode='test')
    else:
        raise NotImplementedError


    dl_test = DataLoader(ds_test,
                        batch_size=n_img_per_gpu,
                        shuffle=False,
                        num_workers=n_workers,
                        pin_memory=True,
                        drop_last=False)

    """ Set Model of CSGCNet """
    ignore_idx = params["dataset_config"]["ignore_idx"]
    base_path_pretrained = Path("/home/wuyi/AnsonD/CSGCNet/scripts/core/models/pretrained_backbones")
    backbone_weights = (base_path_pretrained / params["training_config"]["backbone_weights"]).resolve()
    net = CSGCNet(n_classes=n_classes, backbone_weights=backbone_weights)
    j = 0
    it = 2000
    # for j in range(2):

    state_name = "UDD6_CSGCNet_1024x1024" + f"_iter_{it}.pth"

    #ignore_idx = params["dataset_config"]["ignore_idx"]
    base_path_pretrained = Path("/home/wuyi/AnsonD/CSGCNet/UDD6_CSGCNet")
    backbone_weights = (base_path_pretrained / state_name).resolve()
    net.load_state_dict(torch.load(backbone_weights))
    net.cuda()
    net.eval()

    iou_sum = np.zeros((6,))
    f1_sum = np.zeros((6,))
    acc_sum = np.zeros((6,))
    for index, batch in enumerate(dl_test):
        # if index % 100 == 0:
        #     print(index)
        #     print('%d processd' % (index))
        index = str(index)
        print(index)
        image, label = batch
        # label1 = str(label)
        # print(index)
        # m = m + 1

        with torch.no_grad():  # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
            imgs = Variable(image).cuda()               # (shape: (batch_size, 3, img_h, img_w))
            label = Variable(label).cuda()

            outputs= net(imgs)[1]               # (shape: (batch_size, num_classes, img_h, img_w))

        output = outputs.max(1)[1]
        label = torch.squeeze(label, dim=1)

        iou_sum += iou(output, label, n_classes=6)
        f1_sum += f1(output, label, n_classes=6)
        acc_sum += acc(output, label, n_classes=6)

        outputs = outputs.data.cpu().numpy()  # (shape: (batch_size, num_classes, img_h, img_w))
        pred_label_imgs = np.argmax(outputs, axis=1)  # (shape: (batch_size, img_h, img_w))
        pred_label_imgs = pred_label_imgs.astype(np.uint8)

        for i in range(pred_label_imgs.shape[0]):
            dl_test_path = os.listdir(ds_test.img_dir)
            dl_test_path.sort()
            dl_test_img = dl_test_path[j]
            dl_test_img, _ = os.path.splitext(dl_test_img)
            t = str(dl_test_img)
            print(t)

            mask = get_color_pallete_udd6(pred_label_imgs[i], dataset="udd6")
            mask.save("/home/wuyi/AnsonD/CSGCNet/Output_RGB_UDD6_CSGCNet" + "/" + t + ".png")
            j = j + 1

    class_iou = iou_sum / len(dl_test)
    class_f1 = f1_sum / len(dl_test)
    class_acc = acc_sum / len(dl_test)

    mIoU = np.mean(class_iou)
    mF1 = np.mean(class_f1)
    mACC = np.mean(class_acc)
    print('Val result: mIoU/mF1/mAcc  {:.5f}/{:.5f}/{:.5f}'.format(mIoU, mF1, mACC))
    for i in range(n_classes):
        print("Class_{} Result: class_iou/class_f1/class_acc  {:.5f}/{:.5f}/{:.5f}".format(i, class_iou[i],
                                                                                           class_f1[i],
                                                                                           class_acc[i]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                    type=str,
                    default="/home/wuyi/AnsonD/CSGCNet/configs/train_UDD6.json",)
    args = parser.parse_args()
    test(args.config)