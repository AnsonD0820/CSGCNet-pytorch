import argparse
import datetime
import json
import logging
import time
from pathlib import Path
from shutil import copyfile
from core.models.Adan.adan import Adan
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from core.datasets.potsdam import potsdam
from core.datasets.uavid import UAVid
from core.datasets.udd6 import UDD6
from core.models.CSGCNet import CSGCNet
from core.utils.logger import setup_logger
from core.utils.loss import OhemCELoss, CriterionOhemDSN
from evaluate import MscEval
import os
import json


def train_and_evaluate(config, logger):
    """ Set directories """
    with open(config, "r") as f:
        params = json.loads(f.read())
    respth = Path(params["training_config"]["experiments_path"])
    Path.mkdir(respth, parents=True, exist_ok=True)
    writer = SummaryWriter(respth)

    torch.cuda.set_device(params["training_config"]["gpu_id"])
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:33271',
        world_size=torch.cuda.device_count(),
        rank=params["training_config"]["gpu_id"]
    )
    setup_logger(respth)
    torch.cuda.synchronize()

    """ Set Dataset Params """
    n_classes = params["dataset_config"]["num_classes"]
    n_img_per_gpu = params["training_config"]["batch_size"]
    n_workers = params["training_config"]["num_workers"]
    cropsize = params["dataset_config"]["cropsize"]
    val_batch_size = params["validation_config"]["batch_size"]

    """ Prepare DataLoader """
    if params["dataset_config"]["name"] == "uavid":
        ds_train = UAVid(params, mode='train')
        ds_val = UAVid(params, mode='val')
    elif params["dataset_config"]["name"] == "UDD":
        ds_train = UDD6(params, mode='train')
        ds_val = UDD6(params, mode='val')
    elif params["dataset_config"]["name"] == "potsdam":
        ds_train = potsdam(params, mode='train')
        ds_val = potsdam(params, mode='val')
    else:
        raise NotImplementedError
    sampler = torch.utils.data.distributed.DistributedSampler(ds_train)
    dl_train = DataLoader(ds_train,
                          batch_size=n_img_per_gpu,
                          shuffle=False,
                          sampler=sampler,
                          num_workers=n_workers,
                          pin_memory=True,
                          drop_last=True)
    dl_val = DataLoader(ds_val,
                        batch_size=n_img_per_gpu,
                        shuffle=False,
                        num_workers=n_workers,
                        pin_memory=True,
                        drop_last=True)

    """ Set Model of CABiNet """
    ignore_idx = params["dataset_config"]["ignore_idx"]
    base_path_pretrained = Path("/home/wuyi/AnsonD/CSGCNet/scripts/core/models/pretrained_backbones")
    backbone_weights = (base_path_pretrained / params["training_config"]["backbone_weights"]).resolve()
    net = CSGCNet(n_classes=n_classes, backbone_weights=backbone_weights)

    net.cuda()
    net.train()
    net = nn.parallel.DistributedDataParallel(net,
                                              device_ids=[params["training_config"]["gpu_id"], ],
                                              output_device=params["training_config"]["gpu_id"],
                                              find_unused_parameters=True
                                              )
    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
    criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_16 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criterion = CriterionOhemDSN()

    """ Set Optimization Parameters """
    momentum = params["training_config"]["optimizer_momentum"]
    weight_decay = params["training_config"]["optimizer_weight_decay"]
    lr_start = params["training_config"]["optimizer_lr_start"]
    max_iter = params["training_config"]["max_iterations"]
    power = params["training_config"]["optimizer_power"]
    warmup_steps = params["training_config"]["warmup_stemps"]
    warmup_start_lr = params["training_config"]["warmup_start_lr"]

    """optimizer"""
    optim = Adan([{'params': filter(lambda p: p.requires_grad, net.parameters()), 'lr': lr_start}],lr=warmup_start_lr, weight_decay=weight_decay)

    """ Set Train Loop Params """
    msg_iter = params["training_config"]["msg_iterations"]

    save_steps = params["training_config"]["save_iter"]
    best_score = 0.0
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl_train)
    epoch = 0
    logger.info('\n')
    logger.info('====' * 20)
    logger.info('[INFO]: Begining Training of Model ...\n')

    for it in range(max_iter):
        try:
            im, lb = next(diter)
            if not im.size()[0] == n_img_per_gpu: raise StopIteration
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            diter = iter(dl_train)
            im, lb = next(diter)
        im = im.cuda()
        lb = lb.cuda()
        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        out,out16= net(im)
        loss1 = criteria_p(out, lb)
        loss2 = criteria_16(out16, lb)
        loss = loss1 + loss2
        loss.backward()
        optim.step()
        torch.cuda.synchronize()
        loss1.item()
        loss2.item()
        loss.item()
        loss_avg.append(loss.item())
        torch.cuda.empty_cache()
        """ Log Values """
        if (it + 1) % 200 == 0:
            loss_tensorboard = sum(loss_avg) / len(loss_avg)
            lr = warmup_start_lr
            msg = ''.join([
                'it: {it}/{max_it} || ',
                'lr: {lr:4f} || ',
                'loss: {loss:.4f} || ',
            ]).format(
                it=it + 1,
                max_it=max_iter,
                lr=lr,
                loss=loss_tensorboard,
            )
            logger.info(msg)
            writer.add_scalar('Train/Loss_avg', loss_tensorboard, global_step=(it+1))

        if (it + 1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = warmup_start_lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ''.join([
                'it: {it}/{max_it} || ',
                'lr: {lr:4f} || ',
                'loss: {loss:.4f} || ',
                'eta: {eta} || ',
                'time: {time:.4f}',
            ]).format(
                it=it + 1,
                max_it=max_iter,
                lr=lr,
                loss=loss_avg,
                time=t_intv,
                eta=eta,
            )
            logger.info(msg)
            loss_avg = []
            st = ed

        if (it + 1) % save_steps == 0:
            save_name = params["training_config"]["model_save_name"].split(".pth")[0] + f"_iter_{it + 1}.pth"
            save_pth = respth / save_name
            net.cpu()
            state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            if dist.get_rank() == 0: torch.save(state, str(save_pth))
            logger.info(f'[INFO]: {it + 1} iterations Finished!; Model Saved to: {save_pth}')
            with torch.no_grad():

                net.cuda()
                net.eval()
                torch.cuda.synchronize()
                evaluator = MscEval(net, dl_val, params)
                current_score = evaluator.evaluate()
                if current_score > best_score:
                    save_name = params["training_config"]["model_save_name"].split(".pth")[
                                    0] + f"_iter_{it + 1}_best_mIOU_{current_score:.4f}.pth"
                    save_pth = respth / save_name
                    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                    if dist.get_rank() == 0: torch.save(state, str(save_pth))
                    print(f"[INFO]: mIOU imporved from {best_score:.4f} to {current_score:.4f}")
                    best_score = current_score
                else:
                    print(f"[INFO]: mIOU did not improve from {best_score:.4f}")
                torch.cuda.empty_cache()
                net.cuda()
                net.train()
                torch.cuda.synchronize()
    writer.close()

    """ Dump and Save the Final Model """
    print(f"[INFO]: Epochs Completed {epoch}")
    save_pth = respth / params["training_config"]["model_save_name"]
    net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    if dist.get_rank() == 0: torch.save(state, str(save_pth))
    logger.info('Training Finished!; Model Saved to: {}'.format(save_pth))
    torch.cuda.empty_cache()

    """ Save the Config Files with Experiment """
    config_file_out = respth / config
    dir_path = os.path.dirname(config_file_out)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    copyfile(config, config_file_out)
    p = Path(".")
    file_list = list(p.glob('**/*.py'))
    for file in file_list:
        file_out = respth / file

    file_path = os.path.dirname(file_out)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    copyfile(str(file), str(file_out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default="configs/train_potsdam.json", )
    args = parser.parse_args()
    logger = logging.getLogger()
    train_and_evaluate(args.config, logger)