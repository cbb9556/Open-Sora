import collections
import random
from typing import Optional

import numpy as np
import torch
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _get_default_group
from torch.utils.data import DataLoader

from .datasets import BatchFeatureDataset, VariableVideoTextDataset, VideoTextDataset
from .sampler import BatchDistributedSampler, StatefulDistributedSampler, VariableVideoBatchSampler


# Deterministic dataloader
def get_seed_worker(seed):
    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


def prepare_dataloader( #对不同的数据集类型，返回不同的DataLoader和Sampler
    dataset,
    batch_size=None,
    shuffle=False,
    seed=1024,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    process_group: Optional[ProcessGroup] = None,
    bucket_config=None,
    num_bucket_build_workers=1,
    prefetch_factor=None,
    **kwargs,
):
    _kwargs = kwargs.copy()
    if isinstance(dataset, VariableVideoTextDataset): #不同是视频和文档对齐的数据集
        batch_sampler = VariableVideoBatchSampler( #视频批量采样器
            dataset,
            bucket_config,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            verbose=True,
            num_bucket_build_workers=num_bucket_build_workers,
        )
        return (
            # 初始化DataLoader以用于数据加载
            DataLoader(
                dataset,  # 数据集，包含了样本和它们的标签
                batch_sampler=batch_sampler,  # 定义如何对数据集进行批量采样
                worker_init_fn=get_seed_worker(seed),  # 为每个工作线程设置初始化函数，确保数据加载的随机性在多线程间一致
                pin_memory=pin_memory,  # 如果True，将数据加载到 pinned memory 中，可以加快数据从CPU传输到GPU的速度
                num_workers=num_workers,  # 使用多线程来加载数据，指定线程数量
                collate_fn=collate_fn_default,  # 指定如何将样本组成batch，默认处理是将样本堆叠起来
                prefetch_factor=prefetch_factor,  # 每个worker预取样本的数量，可以提高数据加载效率
                **_kwargs,  # 其他传递给DataLoader的参数，提供了灵活性和可扩展性
            ),
            batch_sampler,
        )
    elif isinstance(dataset, VideoTextDataset):
        process_group = process_group or _get_default_group()
        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            shuffle=shuffle,
        )
        return (
            DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                worker_init_fn=get_seed_worker(seed),
                drop_last=drop_last,
                pin_memory=pin_memory,
                num_workers=num_workers,
                collate_fn=collate_fn_default,
                prefetch_factor=prefetch_factor,
                **_kwargs,
            ),
            sampler,
        )
    elif isinstance(dataset, BatchFeatureDataset):
        sampler = BatchDistributedSampler(
            dataset,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
        )
        return (
            DataLoader(
                dataset,
                batch_size=1,
                sampler=sampler,
                worker_init_fn=get_seed_worker(seed),
                pin_memory=pin_memory,
                num_workers=num_workers,
                collate_fn=collate_fn_batch,
                prefetch_factor=prefetch_factor,
                **_kwargs,
            ),
            sampler,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")


def collate_fn_default(batch):
    # filter out None
    batch = [x for x in batch if x is not None]

    # HACK: for loading text features
    use_mask = False
    if "mask" in batch[0] and isinstance(batch[0]["mask"], int):
        masks = [x.pop("mask") for x in batch]

        texts = [x.pop("text") for x in batch]
        texts = torch.cat(texts, dim=1)
        use_mask = True

    ret = torch.utils.data.default_collate(batch)

    if use_mask:
        ret["mask"] = masks
        ret["text"] = texts
    return ret


def collate_fn_batch(batch):
    """
    Used only with BatchDistributedSampler
    """
    # filter out None
    batch = [x for x in batch if x is not None]
    
    res = torch.utils.data.default_collate(batch)

    # squeeze the first dimension, which is due to torch.stack() in default_collate()
    if isinstance(res, collections.abc.Mapping):
        for k, v in res.items():
            if isinstance(v, torch.Tensor):
                res[k] = v.squeeze(0)
    elif isinstance(res, collections.abc.Sequence):
        res = [x.squeeze(0) if isinstance(x, torch.Tensor) else x for x in res]
    elif isinstance(res, torch.Tensor):
        res = res.squeeze(0)
    else:
        raise TypeError

    return res
