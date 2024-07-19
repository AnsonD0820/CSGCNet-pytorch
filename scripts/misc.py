import numpy as np
from PIL import Image
import torch
from torch import nn


def get_color_pallete_isprs(npimg, dataset='voc'):
    out_img = Image.fromarray(npimg.astype('uint8')).convert('P')
    if dataset == 'potsdam':
        potsdampallete = [
            255, 0, 0,
            255, 255, 255,
            255, 255, 0,
            0, 255, 0,
            0, 255, 255,
            0, 0, 255,
        ]
        out_img.putpalette(potsdampallete)
    else:
        vocpallete = _getvocpallete(256)
        out_img.putpalette(vocpallete)
    return out_img


def get_color_pallete_udd6(npimg, dataset='voc'):
    out_img = Image.fromarray(npimg.astype('uint8')).convert('P')
    if dataset == 'udd6':
        udd6pallete = [
            0, 0, 0,
            102, 102, 156,
            128, 64, 128,
            107, 142, 35,
            0, 0, 142,
            70, 70, 70,
        ]
        out_img.putpalette(udd6pallete)
    else:
        vocpallete = _getvocpallete(256)
        out_img.putpalette(vocpallete)
    return out_img


def _getvocpallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while (lab > 0):
            pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return pallete


def tp(output, target, n_classes=6):
    res = []
    for cls in range(n_classes):
        pred_inds = output == cls
        target_inds = target == cls
        res.append(float(pred_inds[target_inds].sum()))
    return np.array(res).astype(float)


def fp(output, target, n_classes=6):
    res = []
    for cls in range(n_classes):
        pred_inds = output == cls
        target_inds = target != cls
        res.append(float(pred_inds[target_inds].sum()))
    return np.array(res).astype(float)


def fn(output, target, n_classes=6):
    res = []
    for cls in range(n_classes):
        pred_inds = output != cls
        target_inds = target == cls
        res.append(float(pred_inds[target_inds].sum()))
    return np.array(res).astype(float)


def tn(output, target, n_classes=6):
    res = []
    for cls in range(n_classes):
        pred_inds = output != cls
        target_inds = target != cls
        res.append(float(pred_inds[target_inds].sum()))
    return np.array(res).astype(float)


def iou(output, target, n_classes=6):
    smooth = 1e-5
    ious = []
    for cls in range(n_classes):
        pred_inds = output == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        ious.append((float(intersection)+smooth)/ (float(union) + smooth))
    return np.array(ious)*100


def f1(output, target, n_classes=6):
    smooth = 1e-5
    f1 = (2*tp(output, target, n_classes) + smooth)/(2*tp(output, target, n_classes)+fp(output, target, n_classes)+fn(output, target, n_classes) + smooth)
    return f1*100


def acc(output, target, n_classes=6):
    smooth = 1e-5
    acc = (tp(output, target, n_classes) + tn(output, target, n_classes) + smooth)/(tp(output, target, n_classes)+fp(output, target, n_classes) + tn(output, target, n_classes) + fn(output, target, n_classes) + smooth)
    return acc*100