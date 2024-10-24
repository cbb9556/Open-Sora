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
    """
    创建并返回一个用于设置随机种子的工作者函数。

    本函数的目的是确保在分布式计算或并行处理过程中，每个工作者（如数据加载器）能够
    使用相同或不同的随机种子，从而保证实验的可重复性。

    参数:
    seed (int): 用于初始化随机数生成器的种子值。确保所有随机操作（如数据增强、数据采样等）
                在每次运行时保持一致的行为。

    返回:
    seed_worker: 一个内部函数，它接受一个工作者ID作为参数，并使用给定的种子值初始化
                 NumPy、PyTorch和Python的随机数生成器。
    """

    def seed_worker(worker_id):
        """
        设置工作者的随机种子。

        本函数确保每个工作者在处理数据时使用确定的随机种子，从而保证了结果的可重复性。
        它会将传入的种子值用于NumPy、PyTorch和Python的随机数生成器的初始化。

        参数:
        worker_id (int): 工作者的唯一标识符。在分布式计算或并行处理中，每个工作者会分配到
                         一个唯一的ID，用于确保每个工作者的行为是独立且可重复的。
        """
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

# 将掩码和文本分离开，存入到字典中
def collate_fn_default(batch):
    """
    自定义批处理函数，用于处理数据批次。
    该函数处理一个数据批次，移除其中的None元素，并可选地处理文本特征和掩码。
    如果批次包含文本特征，它们将被拼接在一起；如果存在掩码信息，也会进行处理。

    参数:
    - batch (list): 包含一批数据元素的列表，可能包括None元素或带有文本和掩码信息的字典。

    返回:
    - ret (dict 或 tuple): 处理后的数据批次，将掩码和文本分离开，存入到字典中。
    """
    # 过滤掉None元素
    batch = [x for x in batch if x is not None]

    # HACK: 用于加载文本特征
    use_mask = False
    if "mask" in batch[0] and isinstance(batch[0]["mask"], int):
        # 提取并移除掩码信息
        masks = [x.pop("mask") for x in batch]

        # 提取并拼接文本特征
        texts = [x.pop("text") for x in batch]
        texts = torch.cat(texts, dim=1)
        use_mask = True

    # 使用默认的批处理函数处理剩余的数据
    ret = torch.utils.data.default_collate(batch)

    if use_mask:
        # 将掩码和文本特征添加到返回结果中
        ret["mask"] = masks
        ret["text"] = texts
    return ret


def collate_fn_batch(batch):
    """
    仅与 BatchDistributedSampler 一起使用
    该函数用于处理数据批次，主要用于处理数据可能缺失（None）的情况，并根据需要调整批次数据的结构。
    它以 PyTorch 的默认数据合并函数为基础，但在返回处理后的数据批次之前执行额外的处理步骤。
    参数:
    batch (list): 包含一批数据的列表，其中可能包括需要过滤掉的 None 项。

    返回:
    res: 处理后的数据批次，其结构取决于输入数据的结构，如字典、列表或张量。
    """
    # 过滤掉 None
    batch = [x for x in batch if x is not None]

    # 使用默认的 collate 函数将批次合并成一个张量或张量结构
    # 使用PyTorch提供的default_collate函数，将数据批(batch)转换为张量(tensor)格式
    # 这一步是数据预处理中的关键步骤，确保数据能够以正确的格式被模型处理
    res = torch.utils.data.default_collate(batch)

    # 压缩默认 collate 函数中由于 torch.stack() 而产生的第一个维度
    # 根据res的类型来处理结果
    if isinstance(res, collections.abc.Mapping):
        # 如果res是一个字典，遍历其键值对
        for k, v in res.items():
            # 如果值是torch.Tensor类型，对其进行squeeze操作
            if isinstance(v, torch.Tensor):
                res[k] = v.squeeze(0)
    elif isinstance(res, collections.abc.Sequence):
        # 如果res是一个序列，对每个元素进行处理
        res = [x.squeeze(0) if isinstance(x, torch.Tensor) else x for x in res]
    elif isinstance(res, torch.Tensor):
        # 如果res是一个torch.Tensor，直接进行squeeze操作
        res = res.squeeze(0)
    else:
        # 如果res的类型不符合预期，抛出TypeError
        raise TypeError

    return res
