from collections import OrderedDict

import numpy as np

from opensora.utils.misc import get_logger

from .aspect import ASPECT_RATIOS, get_closest_ratio


def find_approximate_hw(hw, hw_dict, approx=0.8):
    for k, v in hw_dict.items():
        if hw >= v * approx:
            return k
    return None


def find_closet_smaller_bucket(t, t_dict, frame_interval):
    # process image
    if t == 1:
        if 1 in t_dict:
            return 1
        else:
            return None
    # process video
    for k, v in t_dict.items():
        if t >= v * frame_interval and v != 1:
            return k
    return None


class Bucket:
    def __init__(self, bucket_config):
        for key in bucket_config:
            assert key in ASPECT_RATIOS, f"Aspect ratio {key} not found."
        # wrap config with OrderedDict
        bucket_probs = OrderedDict()
        bucket_bs = OrderedDict()
        bucket_names = sorted(bucket_config.keys(), key=lambda x: ASPECT_RATIOS[x][0], reverse=True)
        for key in bucket_names:
            bucket_time_names = sorted(bucket_config[key].keys(), key=lambda x: x, reverse=True)
            bucket_probs[key] = OrderedDict({k: bucket_config[key][k][0] for k in bucket_time_names})
            bucket_bs[key] = OrderedDict({k: bucket_config[key][k][1] for k in bucket_time_names})

        # first level: HW
        num_bucket = 0
        hw_criteria = dict()
        t_criteria = dict()
        ar_criteria = dict()
        bucket_id = OrderedDict()
        bucket_id_cnt = 0
        for k1, v1 in bucket_probs.items():
            hw_criteria[k1] = ASPECT_RATIOS[k1][0]
            t_criteria[k1] = dict()
            ar_criteria[k1] = dict()
            bucket_id[k1] = dict()
            for k2, _ in v1.items():
                t_criteria[k1][k2] = k2
                bucket_id[k1][k2] = bucket_id_cnt
                bucket_id_cnt += 1
                ar_criteria[k1][k2] = dict()
                for k3, v3 in ASPECT_RATIOS[k1][1].items():
                    ar_criteria[k1][k2][k3] = v3
                    num_bucket += 1

        # 初始化模型的超参数和配置
        self.bucket_probs = bucket_probs  # 不同bucket的概率分布
        self.bucket_bs = bucket_bs  # 每个bucket的batch size
        self.bucket_id = bucket_id  # bucket的标识符
        self.hw_criteria = hw_criteria  # 高宽比的判断标准
        self.t_criteria = t_criteria  # 类型判断的标准
        self.ar_criteria = ar_criteria  # 面积比的判断标准
        self.num_bucket = num_bucket  # bucket的总数

        get_logger().info("Number of buckets: %s", num_bucket)

    def get_bucket_id(self, T, H, W, frame_interval=1, seed=None):
        """
        根据媒体的尺寸（图像或视频）、帧数和帧间隔确定并返回桶ID。

        参数:
        - T: 视频中的总帧数。对于图像，此值为1。
        - H: 帧的高度。
        - W: 帧的宽度。
        - frame_interval: 视频的帧间隔。默认为1，表示每帧依次考虑。
        - seed: 随机数生成器的种子。确保随机性的可重复性。默认为None。

        返回:
        - 桶ID (元组): 包含三个元素，分别表示硬件ID、时间ID和纵横比ID。
                       如果没有找到合适的桶，返回None。
        """
        # 计算总分辨率
        resolution = H * W
        # 定义一个用于分辨率比较的近似因子
        approx = 0.8

        # 初始化失败标志为True，表示尚未找到合适的桶
        fail = True
        # 遍历每个桶以找到合适的桶
        for hw_id, t_criteria in self.bucket_probs.items():
            # 跳过不满足分辨率要求的桶
            if resolution < self.hw_criteria[hw_id] * approx:
                continue
            # 每个桶的定义， 桶ID，分辨率， 帧长度， 分辨率， 保留概率，当前桶的batch-size
            # 如果样本是图像
            if T == 1:
                # 检查当前桶是否允许单帧样本
                if 1 in t_criteria:
                    # 使用种子初始化随机数生成器以确保可重复性
                    rng = np.random.default_rng(seed + self.bucket_id[hw_id][1])
                    # 根据概率确定样本是否应放入当前桶
                    if rng.random() < t_criteria[1]:
                        # 如果概率满足条件，设置失败标志为False，并记录时间ID
                        fail = False
                        t_id = 1
                        break
                else:
                    # 如果当前桶不允许单帧样本，继续下一个桶
                    continue

            # 否则，查找适合视频的时间ID
            t_fail = True
            for t_id, prob in t_criteria.items():
                # 使用种子初始化随机数生成器以确保可重复性
                rng = np.random.default_rng(seed + self.bucket_id[hw_id][t_id])
                # 处理概率为元组的情况
                if isinstance(prob, tuple):
                    prob_t = prob[1]
                    if rng.random() > prob_t:
                        continue
                # 检查帧数是否大于时间ID乘以帧间隔且时间ID不为1
                if T > t_id * frame_interval and t_id != 1:
                    t_fail = False
                    break
            if t_fail:
                continue

            # 如果概率足够高，离开循环
            if isinstance(prob, tuple):
                prob = prob[0]
            if prob >= 1 or rng.random() < prob:
                fail = False
                break
        if fail:
            return None

        # 获取纵横比ID
        ar_criteria = self.ar_criteria[hw_id][t_id]
        ar_id = get_closest_ratio(H, W, ar_criteria)
        return hw_id, t_id, ar_id

    def get_thw(self, bucket_id):
        assert len(bucket_id) == 3
        T = self.t_criteria[bucket_id[0]][bucket_id[1]]
        H, W = self.ar_criteria[bucket_id[0]][bucket_id[1]][bucket_id[2]]
        return T, H, W

    def get_prob(self, bucket_id):
        return self.bucket_probs[bucket_id[0]][bucket_id[1]]

    def get_batch_size(self, bucket_id):
        return self.bucket_bs[bucket_id[0]][bucket_id[1]]

    def __len__(self):
        return self.num_bucket


def closet_smaller_bucket(value, bucket):
    for i in range(1, len(bucket)):
        if value < bucket[i]:
            return bucket[i - 1]
    return bucket[-1]
