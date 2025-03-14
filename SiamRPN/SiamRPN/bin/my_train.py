import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import numpy as np
import pandas as pd
import os
import cv2
import pickle

import torch.nn as nn
import time

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from collections import OrderedDict
import lmdb
import setproctitle
import argparse

import sys
sys.path.append(os.getcwd())

from IPython import embed

from siamrpn.config import config
from siamrpn.network import SiamRPNNet
#from .dataset import ImagnetVIDDataset 
from got10k.datasets import  GOT10k
from siamrpn.dataset import GOT10kDataset
from siamrpn.transforms import Normalize, ToTensor, RandomStretch, RandomCrop, CenterCrop, RandomBlur, ColorAug
from siamrpn.loss import rpn_smoothL1, rpn_cross_entropy_balance
from siamrpn.visual import visual
from siamrpn.utils import get_topk_box, add_box_img, compute_iou, box_transform_inv, adjust_learning_rate,freeze_layers

from IPython import embed

torch.manual_seed(config.seed)

def train(data_dir, resume_path=None, vis_port=None, init=None):

    #-----------------------
    name='GOT-10k'
    seq_dataset_train= GOT10k(data_dir, subset='train')
    #seq_dataset_val = GOT10k(data_dir, subset='val')
    print('seq_dataset_train', len(seq_dataset_train))  # train-9335 个文件 
   
    # define transforms 
    train_z_transforms = transforms.Compose([
        ToTensor()
    ])
    train_x_transforms = transforms.Compose([
        ToTensor()
    ])
    valid_z_transforms = transforms.Compose([
        ToTensor()
    ])
    valid_x_transforms = transforms.Compose([
        ToTensor()
    ])

    # create dataset 
    # -----------------------------------------------------------------------------------------------------
    # train_dataset = ImagnetVIDDataset(db, train_videos, data_dir, train_z_transforms, train_x_transforms)
    train_dataset  = GOT10kDataset(
        seq_dataset_train, train_z_transforms, train_x_transforms, name)
  
    # valid_dataset  = GOT10kDataset(
    #     seq_dataset_val, valid_z_transforms, valid_x_transforms, name)
   
    anchors = train_dataset.anchors

    # create dataloader
    
    trainloader = DataLoader(  dataset    = train_dataset,
                                batch_size = config.train_batch_size,
                                shuffle    = True, num_workers= config.train_num_workers,
                                pin_memory = True,drop_last =True)
                                
    # validloader = DataLoader(dataset = valid_dataset, batch_size=config.valid_batch_size ,
    #                          shuffle=False, pin_memory=True,
    #                          num_workers=config.valid_num_workers, drop_last=True)
    
    # create summary writer
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    summary_writer = SummaryWriter(config.log_dir)
    if vis_port:
        vis = visual(port=vis_port)

    # start training
    # -----------------------------------------------------------------------------------------------------#
    model = SiamRPNNet()

    model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,momentum=config.momentum, weight_decay=config.weight_decay)
  
    #load model weight
    # -----------------------------------------------------------------------------------------------------#
    start_epoch = 1
    if resume_path and init: #不加载optimizer
        print("init training with checkpoint %s" % resume_path + '\n')
        print('------------------------------------------------------------------------------------------------ \n')
        checkpoint = torch.load(resume_path)
        if 'model' in checkpoint.keys():
            model.load_state_dict(checkpoint['model'])
        else:
            model_dict = model.state_dict()#获取网络参数
            model_dict.update(checkpoint)#更新网络参数
            model.load_state_dict(model_dict)#加载网络参数
        del checkpoint
        torch.cuda.empty_cache()#清空缓存
        print("inited checkpoint")
    elif resume_path and not init: #获取某一个checkpoint恢复训练
        print("loading checkpoint %s" % resume_path + '\n')
        print('------------------------------------------------------------------------------------------------ \n')
        checkpoint = torch.load(resume_path)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            model.load_state_dict(checkpoint)

        del checkpoint
        torch.cuda.empty_cache()  #缓存清零
        print("loaded checkpoint")
    elif not resume_path and config.pretrained_model: #加载预习训练模型
        print("loading pretrained model %s" % config.pretrained_model + '\n')
        print('------------------------------------------------------------------------------------------------ \n')
        checkpoint = torch.load(config.pretrained_model)
        # change name and load parameters
        checkpoint = {k.replace('features.features', 'featureExtract'): v for k, v in checkpoint.items()}
        model_dict = model.state_dict()
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)

    # print(model.featureExtract[:10])
    #如果有多块GPU，则开启多GPU模式
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    for epoch in range(start_epoch, config.epoch + 1):
        train_loss = []
        model.train() # 训练模式

        # if config.fix_former_3_layers: #True，固定模型的前10层参数不变
        #     我的修改,第10个epoch开始放开训练 ,对最终的结果并没有提升
        #     if epoch<10:
        #         if torch.cuda.device_count() > 1: #多GPU
        #             freeze_layers(model.module) 
        #         else: # 单GPU
        #             freeze_layers(model)
        
        #True，固定模型的前10层参数不变
        if config.fix_former_3_layers: 
            if torch.cuda.device_count() > 1: #多GPU
                freeze_layers(model.module) 
            else: # 单GPU
                freeze_layers(model)

        loss_temp_cls = 0
        loss_temp_reg = 0
        for i, data in enumerate(tqdm(trainloader)):
            exemplar_imgs, instance_imgs, regression_target, conf_target = data#Exemplar_imgs模板图像, Instance_imgs搜索图像, Regression_target bbox-groundtruth的偏移, Conf_target 根据iou计算的正负样本
            # conf_target (8,1125) (8,225x5)

            regression_target, conf_target = regression_target.cuda(), conf_target.cuda()
            #pre_score=batchsize,10,19,19 ； pre_regression=[batchsize,20,19,19]
            pred_score, pred_regression = model(exemplar_imgs.cuda(), instance_imgs.cuda())
            # [batchsize, 5x19x19, 2]=[batchsize,1805,2]
            pred_conf = pred_score.reshape(-1, 2, config.anchor_num * config.score_size * config.score_size).permute(0,
                                                                                                                     2,
                                                                                                                     1)
             #[batchsize,5x19x19,4] =[batchsize,1805,4]
            pred_offset = pred_regression.reshape(-1, 4,
                                                  config.anchor_num * config.score_size * config.score_size).permute(0,
                                                                                                                     2,
                                                                                                                     1)
            cls_loss = rpn_cross_entropy_balance(pred_conf, conf_target, config.num_pos, config.num_neg, anchors,
                                                 ohem_pos=config.ohem_pos, ohem_neg=config.ohem_neg)
            reg_loss = rpn_smoothL1(pred_offset, regression_target, conf_target, config.num_pos, ohem=config.ohem_reg)
            loss = cls_loss + config.lamb * reg_loss #分类权重和回归权重
            optimizer.zero_grad()#梯度
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)#config.clip=10 ，clip_grad_norm_梯度裁剪，防止梯度爆炸
            optimizer.step()

            step = (epoch - 1) * len(trainloader) + i
            summary_writer.add_scalar('train/cls_loss', cls_loss.data, step)
            summary_writer.add_scalar('train/reg_loss', reg_loss.data, step)
            train_loss.append(loss.detach().cpu())#当前计算图中分离下来的，但是仍指向原变量的存放位置,requires_grad=false
            loss_temp_cls += cls_loss.detach().cpu().numpy()
            loss_temp_reg += reg_loss.detach().cpu().numpy()
            # if vis_port:
            #     vis.plot_error({'rpn_cls_loss': cls_loss.detach().cpu().numpy().ravel()[0],
            #                     'rpn_regress_loss': reg_loss.detach().cpu().numpy().ravel()[0]}, win=0)
            if (i + 1) % config.show_interval == 0:
            #if (i + 1) % 5 == 0:
                tqdm.write("[epoch %2d][iter %4d] cls_loss: %.4f, reg_loss: %.4f lr: %.2e"
                           % (epoch, i, loss_temp_cls / config.show_interval, loss_temp_reg / config.show_interval,
                              optimizer.param_groups[0]['lr']))
                loss_temp_cls = 0
                loss_temp_reg = 0

        train_loss = np.mean(train_loss)


        valid_loss=0

        print("EPOCH %d valid_loss: %.4f, train_loss: %.4f" % (epoch, valid_loss, train_loss))
        
        summary_writer.add_scalar('valid/loss',valid_loss, (epoch + 1) * len(trainloader))
        
        adjust_learning_rate(optimizer,config.gamma)  # adjust before save, and it will be epoch+1's lr when next load
       
        if epoch % config.save_interval == 0:
            if not os.path.exists('../models/'):
                os.makedirs("../models/")
            save_name = "../models/siamrpn_{}.pth".format(epoch)
            #new_state_dict = model.state_dict()
            if torch.cuda.device_count() > 1: # 多GPU训练
                new_state_dict=model.module.state_dict()
            else:  #单GPU训练
                new_state_dict=model.state_dict()
            torch.save({
                'epoch': epoch,
                'model': new_state_dict,
                'optimizer': optimizer.state_dict(),
            }, save_name)
            print('save model: {}'.format(save_name))


os.environ["CUDA_VISIBLE_DEVICES"] = "0" #多卡情况下默认多卡训练,如果想单卡训练,设置为"0"

if __name__ == '__main__':
    
    # 参 数
    parser=argparse.ArgumentParser(description=" SiamRPN Train")

    parser.add_argument('--resume_path',default='', type=str, help=" input gpu id ") # resume_path 为空, 默认加载预训练模型alexnet,在config中有配置

    parser.add_argument('--data',default='/home/xyz/data/GOT10K',type=str,help=" the path of data")

    args=parser.parse_args()

    # 训 练 
    train(args.data, args.resume_path)  
