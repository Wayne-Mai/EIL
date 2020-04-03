from torchvision import transforms
from torch.utils.data import DataLoader
from .dataset_cub import CUBClsDataset, CUBCamDataset

import random
import os
import sys

import torch

import torchvision.datasets as datasets
import torchvision.models as models


def data_loader(args):

    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]

    tsfm_train = transforms.Compose([transforms.Resize((args.resize_size, args.resize_size)),
                                     transforms.RandomCrop(args.crop_size),
                                     transforms.CenterCrop(args.crop_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)
                                     ])

    if args.tencrop == 'True':
        # default behavior is to disable tencrop
        func_transforms = [transforms.Resize(args.resize_size),
                           transforms.TenCrop(args.crop_size),
                           transforms.Lambda(
            lambda crops: torch.stack(
                [transforms.Normalize(mean_vals, std_vals)(transforms.ToTensor()(crop)) for crop in crops])),
        ]
    else:
        func_transforms = []

        # print input_size, crop_size
        if args.resize_size == 0 or args.crop_size == 0:
            pass
        else:
            func_transforms.append(transforms.Resize(
                (args.resize_size, args.resize_size)))
            func_transforms.append(transforms.CenterCrop(args.crop_size))
            # func_transforms.append(transforms.Resize(
            #     (256,256)))
            # func_transforms.append(transforms.CenterCrop(224))

        func_transforms.append(transforms.ToTensor())
        func_transforms.append(transforms.Normalize(mean_vals, std_vals))

    tsfm_test = transforms.Compose(func_transforms)

    if args.dataset == 'CUB':
        img_train = CUBClsDataset(
            root=args.data, datalist=args.train_list, transform=tsfm_train, args=args)
        img_test = CUBCamDataset(
            root=args.data, datalist=args.test_list, transform=tsfm_test)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            img_train)
    else:
        train_sampler = None

    train_loader = DataLoader(img_train,
                              batch_size=args.batch_size,
                              shuffle=False, # note 
                              sampler=train_sampler,
                              num_workers=args.workers)

    val_loader = DataLoader(img_test,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.workers)

    return train_loader, val_loader, train_sampler
