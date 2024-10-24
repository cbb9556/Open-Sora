from collections import OrderedDict, defaultdict
from pprint import pformat
from typing import Iterator, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler

from opensora.utils.misc import format_numel_str, get_logger

from .aspect import get_num_pixels
from .bucket import Bucket
from .datasets import VariableVideoTextDataset


# use pandarallel to accelerate bucket processing
# NOTE: pandarallel should only access local variables
def apply(data, method=None, frame_interval=None, seed=None, num_bucket=None):
    # 应用给定的方法get_bucket_id,到数据上，
    return method(
        data["num_frames"], #数据的帧数
        data["height"],
        data["width"],
        frame_interval,
        seed + data["id"] * num_bucket, # 此表达式的目的是为了根据数据id和桶的数量计算一个唯一的seed值
    )


class StatefulDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_index: int = 0

    def __iter__(self) -> Iterator:
        iterator = super().__iter__()
        indices = list(iterator)
        indices = indices[self.start_index :]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def reset(self) -> None:
        self.start_index = 0

    def state_dict(self, step) -> dict:
        return {"start_index": step}

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)


class VariableVideoBatchSampler(DistributedSampler): #继承自分布式采样器
    def __init__(
        self,
        dataset: VariableVideoTextDataset,
        bucket_config: dict,
        num_replicas: Optional[int] = None, # 这表示 num_replicas 参数可以是一个整数，也可以是 None
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        verbose: bool = False,
        num_bucket_build_workers: int = 1,
    ) -> None:
        super().__init__(
            dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last
        )
        self.dataset = dataset
        self.bucket = Bucket(bucket_config)
        self.verbose = verbose
        self.last_micro_batch_access_index = 0
        self.approximate_num_batch = None
        self._get_num_batch_cached_bucket_sample_dict = None
        self.num_bucket_build_workers = num_bucket_build_workers

    def __iter__(self) -> Iterator[List[int]]:
        """
        自定义迭代器方法，生成数据样本的批次，考虑了桶化、洗牌以及必要时的填充或丢弃样本。

        返回:
            Iterator[List[int]]: 迭代器，每次迭代返回一个数据样本批次，每个批次是一个包含样本索引的列表。
        """
        # 确定要使用的桶样本字典，可以是缓存的或新生成的
        if self._get_num_batch_cached_bucket_sample_dict is not None:
            bucket_sample_dict = self._get_num_batch_cached_bucket_sample_dict
            self._get_num_batch_cached_bucket_sample_dict = None
        else:
            bucket_sample_dict = self.group_by_bucket()
            if self.verbose:
                self._print_bucket_info(bucket_sample_dict)

        # 初始化随机数生成器
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        bucket_micro_batch_count = OrderedDict()
        bucket_last_consumed = OrderedDict()

        # 处理样本
        for bucket_id, data_list in bucket_sample_dict.items():
            # 处理 drop_last
            bs_per_gpu = self.bucket.get_batch_size(bucket_id)
            remainder = len(data_list) % bs_per_gpu

            if remainder > 0:
                if not self.drop_last:
                    # 如果有剩余，填充以使其可被整除
                    data_list += data_list[: bs_per_gpu - remainder]
                else:
                    # 直接丢弃剩余部分以使其可被整除
                    data_list = data_list[:-remainder]
            bucket_sample_dict[bucket_id] = data_list

            # 处理洗牌
            if self.shuffle:
                data_indices = torch.randperm(len(data_list), generator=g).tolist()
                data_list = [data_list[i] for i in data_indices]
                bucket_sample_dict[bucket_id] = data_list

            # 计算每个桶的微批次数量
            num_micro_batches = len(data_list) // bs_per_gpu
            bucket_micro_batch_count[bucket_id] = num_micro_batches

        # 计算桶访问顺序
        bucket_id_access_order = []
        for bucket_id, num_micro_batch in bucket_micro_batch_count.items():
            bucket_id_access_order.extend([bucket_id] * num_micro_batch)

        # 随机化访问顺序
        if self.shuffle:
            bucket_id_access_order_indices = torch.randperm(len(bucket_id_access_order), generator=g).tolist()
            bucket_id_access_order = [bucket_id_access_order[i] for i in bucket_id_access_order_indices]

        # 使桶访问次数可被数据并行度整除
        remainder = len(bucket_id_access_order) % self.num_replicas
        if remainder > 0:
            if self.drop_last:
                bucket_id_access_order = bucket_id_access_order[: len(bucket_id_access_order) - remainder]
            else:
                bucket_id_access_order += bucket_id_access_order[: self.num_replicas - remainder]

        # 准备每个批次的数据
        num_iters = len(bucket_id_access_order) // self.num_replicas
        start_iter_idx = self.last_micro_batch_access_index // self.num_replicas

        # 重新计算微批次消耗
        self.last_micro_batch_access_index = start_iter_idx * self.num_replicas
        for i in range(self.last_micro_batch_access_index):
            bucket_id = bucket_id_access_order[i]
            bucket_bs = self.bucket.get_batch_size(bucket_id)
            if bucket_id in bucket_last_consumed:
                bucket_last_consumed[bucket_id] += bucket_bs
            else:
                bucket_last_consumed[bucket_id] = bucket_bs

        # 生成每个迭代的批次
        for i in range(start_iter_idx, num_iters):
            bucket_access_list = bucket_id_access_order[i * self.num_replicas: (i + 1) * self.num_replicas]
            self.last_micro_batch_access_index += self.num_replicas

            # 计算每个访问的数据样本
            bucket_access_boundaries = []
            for bucket_id in bucket_access_list:
                bucket_bs = self.bucket.get_batch_size(bucket_id)
                last_consumed_index = bucket_last_consumed.get(bucket_id, 0)
                bucket_access_boundaries.append([last_consumed_index, last_consumed_index + bucket_bs])

                # 更新消耗
                if bucket_id in bucket_last_consumed:
                    bucket_last_consumed[bucket_id] += bucket_bs
                else:
                    bucket_last_consumed[bucket_id] = bucket_bs

            # 计算每个GPU访问的数据范围
            bucket_id = bucket_access_list[self.rank]
            boundary = bucket_access_boundaries[self.rank]
            cur_micro_batch = bucket_sample_dict[bucket_id][boundary[0]: boundary[1]]

            # 将 t, h, w 编码到样本索引中
            real_t, real_h, real_w = self.bucket.get_thw(bucket_id)
            cur_micro_batch = [f"{idx}-{real_t}-{real_h}-{real_w}" for idx in cur_micro_batch]
            yield cur_micro_batch

        self.reset()

    def __len__(self) -> int: #每个gpu上面处理的批次大小
        return self.get_num_batch() // dist.get_world_size()

    def group_by_bucket(self) -> dict: #pd并行函数，按组分桶
        bucket_sample_dict = OrderedDict()

        from pandarallel import pandarallel

        # 初始化pandarallel库以并行处理数据
        # 设置工作线程数为self.num_bucket_build_workers，不显示进度条
        pandarallel.initialize(nb_workers=self.num_bucket_build_workers, progress_bar=False)

        get_logger().info("Building buckets...")
        # 使用parallel_apply方法对数据集中的每行数据应用指定的函数，这是多线程处理数据的一种方式
        # 目的在于高效地为每个数据项分配一个桶ID
        bucket_ids = self.dataset.data.parallel_apply( # parallel_apply是 pd的库函数，这个函数，需要传入 aplly函数，aplly函数是自己定义的，将apply函数的结果组装成 List返回
            apply,  # method = get_bucket_id 指定应用的函数，这里应该是某种形式的apply函数，用于应用bucket.get_bucket_id方法
            axis=1,  # 指定应用函数的轴，1表示按行应用函数
            method=self.bucket.get_bucket_id,  # 指定具体的方法，这里是获取桶ID的方法
            frame_interval=self.dataset.frame_interval,  # 传递数据集的帧间隔参数，用于方法中可能的帧采样
            seed=self.seed + self.epoch,  # 结合种子和当前纪元数以确保随机性的同时具有可重复性
            num_bucket=self.bucket.num_bucket,  # 传递桶的总数，用于分配桶ID
        ) # 返回并行 处理之后的 bucket_id的 list

        # group by bucket
        # each data sample is put into a bucket with a similar image/video size
        # 处理并行返回的结果， 使用ordereddict 按照顺序排列
        for i in range(len(self.dataset)): #对于 dataset的每条数据，找到其桶，按照顺序入桶，进行映射
            bucket_id = bucket_ids[i]
            if bucket_id is None:
                continue
            if bucket_id not in bucket_sample_dict:
                bucket_sample_dict[bucket_id] = []
            bucket_sample_dict[bucket_id].append(i)
        return bucket_sample_dict

    def get_num_batch(self) -> int:
        """
        计算并返回当前数据的近似批次数量。
        该方法首先按桶对数据进行分组，然后缓存桶信息以供后续使用。
        如果启用了详细模式，则会打印详细的桶信息。
        返回:
            int: 近似的批次数量。
        """
        # 按桶对数据进行分组
        bucket_sample_dict = self.group_by_bucket()
        # 缓存桶信息以供后续使用
        self._get_num_batch_cached_bucket_sample_dict = bucket_sample_dict

        # 计算批次数量
        if self.verbose:
            # 在详细模式下打印桶信息
            self._print_bucket_info(bucket_sample_dict)
        return self.approximate_num_batch #返回桶近似后，批次的大小

    def _print_bucket_info(self, bucket_sample_dict: dict) -> None:
        # collect statistics
        total_samples = 0
        total_batch = 0
        num_aspect_dict = defaultdict(lambda: [0, 0])
        num_hwt_dict = defaultdict(lambda: [0, 0])
        for k, v in bucket_sample_dict.items():
            size = len(v)
            num_batch = size // self.bucket.get_batch_size(k[:-1])

            total_samples += size
            total_batch += num_batch

            num_aspect_dict[k[-1]][0] += size
            num_aspect_dict[k[-1]][1] += num_batch
            num_hwt_dict[k[:-1]][0] += size
            num_hwt_dict[k[:-1]][1] += num_batch

        # sort
        num_aspect_dict = dict(sorted(num_aspect_dict.items(), key=lambda x: x[0]))
        num_hwt_dict = dict(
            sorted(num_hwt_dict.items(), key=lambda x: (get_num_pixels(x[0][0]), x[0][1]), reverse=True)
        )
        num_hwt_img_dict = {k: v for k, v in num_hwt_dict.items() if k[1] == 1}
        num_hwt_vid_dict = {k: v for k, v in num_hwt_dict.items() if k[1] > 1}

        # log
        if dist.get_rank() == 0 and self.verbose:
            get_logger().info("Bucket Info:")
            get_logger().info(
                "Bucket [#sample, #batch] by aspect ratio:\n%s", pformat(num_aspect_dict, sort_dicts=False)
            )
            get_logger().info(
                "Image Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_hwt_img_dict, sort_dicts=False)
            )
            get_logger().info(
                "Video Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_hwt_vid_dict, sort_dicts=False)
            )
            get_logger().info(
                "#training batch: %s, #training sample: %s, #non empty bucket: %s",
                format_numel_str(total_batch),
                format_numel_str(total_samples),
                len(bucket_sample_dict),
            )
        self.approximate_num_batch = total_batch

    def reset(self):
        self.last_micro_batch_access_index = 0

    def state_dict(self, num_steps: int) -> dict:
        # the last_micro_batch_access_index in the __iter__ is often
        # not accurate during multi-workers and data prefetching
        # thus, we need the user to pass the actual steps which have been executed
        # to calculate the correct last_micro_batch_access_index
        return {"seed": self.seed, "epoch": self.epoch, "last_micro_batch_access_index": num_steps * self.num_replicas}

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)


class BatchDistributedSampler(DistributedSampler):
    """
    Used with BatchDataset;
    Suppose len_buffer == 5, num_buffers == 6, #GPUs == 3, then
           | buffer {i}          | buffer {i+1}
    ------ | ------------------- | -------------------
    rank 0 |  0,  1,  2,  3,  4, |  5,  6,  7,  8,  9
    rank 1 | 10, 11, 12, 13, 14, | 15, 16, 17, 18, 19
    rank 2 | 20, 21, 22, 23, 24, | 25, 26, 27, 28, 29
    """

    def __init__(self, dataset: Dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self.start_index = 0

    def __iter__(self):
        num_buffers = self.dataset.num_buffers
        len_buffer = self.dataset.len_buffer
        num_buffers_i = num_buffers // self.num_replicas
        num_samples_i = len_buffer * num_buffers_i

        indices_i = np.arange(self.start_index, num_samples_i) + self.rank * num_samples_i
        indices_i = indices_i.tolist()

        return iter(indices_i)

    def reset(self):
        self.start_index = 0

    def state_dict(self, step) -> dict:
        return {"start_index": step}

    def load_state_dict(self, state_dict: dict):
        self.start_index = state_dict["start_index"] + 1
