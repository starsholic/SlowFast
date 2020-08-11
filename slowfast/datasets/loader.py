#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""

import itertools
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from slowfast.datasets.multigrid_helper import ShortCycleBatchSampler

from .build import build_dataset


import torch
import re
from torch._six import container_abcs, string_classes, int_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')

def mg_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    temporal_shape_batch = [batch[i][0].shape[1] for i in range(len(batch))]
    #padding temporal axis with 0
    
    for elem in batch:
        template_tensor = torch.zeros(3,400,256,339)
        t_shape = elem[0].shape[1]
        template_tensor[:,:t_shape,:,:]

    elem = batch[0]
    # print('elem:',elem) 
    # print('len(elem)',len(elem)) #2  3
    # print('elem[0].shape',elem[0].shape) #torch.Size([3, 9, 256, 339])  torch.Size([9, 256, 339])
    # print('elem[1].shape',elem[1].shape) #torch.Size([3, 39, 256, 339])  torch.Size([9, 256, 339])
    # shape = [elem[i].shape for i in range(len(elem))]
    # print(str(shape))
    # elem2 = batch[1]
    # shape2 = [elem2[i].shape for i in range(len(elem2))]
    # print('shape2:',str(shape2)) #[torch.Size([54, 256, 339]), torch.Size([54, 256, 339]), torch.Size([54, 256, 339])]

    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        # for elem in batch:

        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        # print(transposed[0])
        # temporal_shape_batch = [batch[i][0].shape[1] for i in range(len(batch))]
        # print('temporal_shape_batch',str(temporal_shape_batch))
        
        return [mg_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def detection_collate(batch):
    """
    Collate function for detection task. Concatanate bboxes, labels and
    metadata from different samples in the first dimension instead of
    stacking them to have a batch-size dimension.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated detection data batch.
    """
    
    inputs, labels, video_idx, extra_data = zip(*batch)
    ##inputs tuple , len(inputs)=8 其中一个sample list是inputs[0]，inputs[0]有两个pathway,
    #inputs[0][0].shape  torch.Size([3, 9, 256, 339])  ,inputs[0][1].shape  torch.Size([3, 39, 256, 339])
    # print('inputs in loader.py:',type(inputs))
    # print('len(inputs)',len(inputs))
    # print('inputs[0].shape',len(inputs[0]))
    # print('inputs[0][0].shape',inputs[0][0].shape)
    # print('inputs[0][1].shape',inputs[0][1].shape)
    inputs, video_idx = mg_collate(inputs), default_collate(video_idx)
    labels = torch.tensor(np.concatenate(labels, axis=0)).float()

    collated_extra_data = {}
    for key in extra_data[0].keys():
        data = [d[key] for d in extra_data]
        if key == "boxes" or key == "ori_boxes":
            # Append idx info to the bboxes before concatenating them.
            bboxes = [
                np.concatenate(
                    [np.full((data[i].shape[0], 1), float(i)), data[i]], axis=1
                )
                for i in range(len(data))
            ]
            bboxes = np.concatenate(bboxes, axis=0)
            collated_extra_data[key] = torch.tensor(bboxes).float()
        elif key == "metadata":
            collated_extra_data[key] = torch.tensor(
                list(itertools.chain(*data))
            ).view(-1, 2)
        else:
            collated_extra_data[key] = default_collate(data)

    return inputs, labels, video_idx, collated_extra_data


def construct_loader(cfg, split, is_precise_bn=False):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)

    if cfg.MULTIGRID.SHORT_CYCLE and split in ["train"] and not is_precise_bn:
        # Create a sampler for multi-process training
        sampler = (
            DistributedSampler(dataset)
            if cfg.NUM_GPUS > 1
            else RandomSampler(dataset)
        )
        batch_sampler = ShortCycleBatchSampler(
            sampler, batch_size=batch_size, drop_last=drop_last, cfg=cfg
        )
        # Create a loader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        )
    else:
        # Create a sampler for multi-process training
        sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
        # Create a loader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
            collate_fn=detection_collate if cfg.DETECTION.ENABLE else None,
        )
    return loader


def shuffle_dataset(loader, cur_epoch):
    """"
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    sampler = (
        loader.batch_sampler.sampler
        if isinstance(loader.batch_sampler, ShortCycleBatchSampler)
        else loader.sampler
    )
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)
