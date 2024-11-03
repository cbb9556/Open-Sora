# Define dataset
dataset = dict(
    type="VariableVideoTextDataset",
    data_path=None,
    num_frames=None,
    frame_interval=3,
    image_size=(None, None),
    transform_name="resize_crop",
)
# IMG: 1024 (20%) 512 (30%) 256 (50%) drop (50%)
bucket_config = {  # 1s/it
    "144p": {1: (0.5, 48), 16: (1.0, 6), 32: (1.0, 3), 96: (1.0, 1)},
    "256": {1: (0.5, 24), 16: (0.5, 3), 48: (0.5, 1), 64: (0.0, None)},
    "240p": {16: (0.3, 2), 32: (0.3, 1), 64: (0.0, None)},
    "512": {1: (0.4, 12)},
    "1024": {1: (0.3, 3)},
}

# 在上述代码中，"image_head": 0.025 表示在图像处理过程中，有 2.5% 的概率会替换图像中头部位置的特征。具体来说：
# 图像中头部位置：通常指的是图像的上部区域，可能对应于图像中的某些特定部分，例如人脸的上半部分。
# 特征替换：在这个区域内的某些特征会被其他特征或随机值替换，以实现某种数据增强或遮挡效果。
# 这种技术常用于图像处理和机器学习任务中，通过引入一定的随机性来增加模型的鲁棒性和泛化能力。
# 定义一个字典，用于指定不同掩码类型的比率
mask_ratios = {
    "identity": 0.75,         # 保持原始状态的比率
    "quarter_random": 0.025,  # 随机替换部分特征的比率
    "quarter_head": 0.025,    # 替换前1/4 头部部分特征的比率
    "quarter_tail": 0.025,    # 替换后1/4 尾部部分特征的比率
    "quarter_head_tail": 0.05, # 替换头部和尾部部分特征的比率
    "image_random": 0.025,    # 图像中随机位置特征替换的比率
    "image_head": 0.025,      # 图像中头部位置特征替换的比率
    "image_tail": 0.025,      # 图像中尾部位置特征替换的比率
    "image_head_tail": 0.05,  # 图像中头部和尾部位置特征替换的比率
}

# Define acceleration
num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
grad_checkpoint = False
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="STDiT2-XL/2",
    from_pretrained=None,
    input_sq_size=512,  # pretrained model is trained on 512x512
    qk_norm=True,
    qk_norm_legacy=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
    micro_batch_size=4,
    local_files_only=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=200,
    shardformer=True,
    local_files_only=True,
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)

# Others
seed = 42
outputs = "outputs"
wandb = False

epochs = 1000
log_every = 10
ckpt_every = 500
load = None

batch_size = None
lr = 2e-5
grad_clip = 1.0
